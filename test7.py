import tensorflow as tf
import numpy as np

train_files = ['/work/tvoigtlaender/TauTransformerComparison/dataloader_dev/TauMLTools/data/ShuffleMergeSpectral_333']

# def _return_self(x):
#     return x
# def _return_interleave(x):
#     return x.interleave(_return_self, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
# def _load_datasets(files):
#     _dataset = []
#     for p in files:
#         _dataset.append(
#             tf.data.Dataset.load(
#                 p, 
#                 compression='GZIP',
#                 reader_func=_return_interleave
#                 )
#             )
#     return _dataset


# def _load_datasets(files):
#     _dataset = []
#     for p in files:
#         _dataset.append(
#             tf.data.Dataset.load(
#                 p, 
#                 compression='GZIP',
#                 reader_func=lambda dataset: dataset.interleave(
#                     lambda x: x, 
#                     cycle_length=1, 
#                     num_parallel_calls=tf.data.AUTOTUNE
#                     )
#                 )
#             )
#     return _dataset


# dataset_train = _load_datasets(train_files)
# print(dataset_train)
import tensorflow as tf

def write_tfrecords(dataset, output_path, num_shards=10):
    """
    Write dataset to TFRecord files with sharding.
    Args:
        dataset: tf.data.Dataset with specified element spec
        output_path: Base path for output files (will append -XXXXX for shards)
        num_shards: Number of shards to split data into
    """
    def serialize_ragged(rt):
        """Convert ragged or eager tensor to serializable format."""
        if not isinstance(rt, tf.RaggedTensor):
            rt = tf.RaggedTensor.from_tensor(rt)
        
        # Ensure correct types
        values = rt.values.numpy().astype(np.float32)
        splits = rt.row_splits.numpy().astype(np.int64)
        
        return {
            'values': values,
            'row_splits': splits
        }

    def create_feature(value):
        """Create feature with explicit type handling"""
        if isinstance(value, np.ndarray):
            if value.dtype in [np.int32, np.int64]:
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value.astype(np.int64).flatten())
                )
            return tf.train.Feature(
                float_list=tf.train.FloatList(value=value.astype(np.float32).flatten())
            )
        elif isinstance(value, (int, np.int32, np.int64)):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=[np.int64(value)])
            )
        else:
            return tf.train.Feature(
                float_list=tf.train.FloatList(value=[np.float32(value)])
            )

    def create_example(features):
        """Convert one sample to tf.train.Example"""
        ragged1, ragged2, ragged3, dense1, dense2 = features
        # Convert ragged tensors
        r1 = serialize_ragged(ragged1)
        r2 = serialize_ragged(ragged2)
        r3 = serialize_ragged(ragged3)
        feature = {
            'ragged1_values': create_feature(r1['values']),
            'ragged1_splits': create_feature(r1['row_splits']),
            'ragged2_values': create_feature(r2['values']),
            'ragged2_splits': create_feature(r2['row_splits']),
            'ragged3_values': create_feature(r3['values']),
            'ragged3_splits': create_feature(r3['row_splits']),
            'dense1': create_feature(dense1.numpy()),
            'dense2': create_feature(dense2.numpy())
        }
        return tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
    # Write sharded TFRecord files
    for shard_id in range(num_shards):
        shard_path = f"{output_path}-{shard_id:05d}-of-{num_shards:05d}"
        with tf.io.TFRecordWriter(shard_path) as writer:
            # Take shard of dataset
            shard = dataset.shard(num_shards, shard_id)
            for features in shard:
                example = create_example(features)
                writer.write(example.SerializeToString())

def read_tfrecords(file_pattern):
    """Read TFRecord files with explicit types"""
    feature_description = {
        'ragged1_values': tf.io.VarLenFeature(tf.float32),
        'ragged1_splits': tf.io.VarLenFeature(tf.int64),
        'ragged2_values': tf.io.VarLenFeature(tf.float32),
        'ragged2_splits': tf.io.VarLenFeature(tf.int64),
        'ragged3_values': tf.io.VarLenFeature(tf.float32),
        'ragged3_splits': tf.io.VarLenFeature(tf.int64),
        'dense1': tf.io.FixedLenFeature([43], tf.float32),
        'dense2': tf.io.FixedLenFeature([10], tf.int64),  # Changed from int32 to int64
    }
    def reconstruct_ragged(values, splits):
        """Reconstruct RaggedTensor from values and splits"""
        values = tf.sparse.to_dense(values)
        splits = tf.sparse.to_dense(splits)
        return tf.RaggedTensor.from_row_splits(values, splits)
    def parse_tfrecord(example_proto):
        """Parse TFRecord example into original tensor structure"""
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        # Reconstruct ragged tensors
        ragged1 = reconstruct_ragged(
            parsed['ragged1_values'], parsed['ragged1_splits']
        )
        ragged2 = reconstruct_ragged(
            parsed['ragged2_values'], parsed['ragged2_splits']
        )
        ragged3 = reconstruct_ragged(
            parsed['ragged3_values'], parsed['ragged3_splits']
        )
        # Get dense tensors
        dense1 = parsed['dense1']
        dense2 = parsed['dense2']
        return ragged1, ragged2, ragged3, dense1, dense2
    # Create dataset from TFRecord files
    dataset = tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(file_pattern)
    )
    # Parse examples
    dataset = dataset.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset

# Example usage:


# Example usage:
# write_tfrecords(dataset, "path/to/output/data", num_shards=10)
    
output_filename = "data/tfrecord"
dd = tf.data.Dataset.load(train_files[0], compression="GZIP")
write_tfrecords(dd, output_filename)
dataset = read_tfrecords(f"{output_filename}*")

print(dataset)
print(dd)
print(dd==dataset)
