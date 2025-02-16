import tensorflow as tf


def serialize_example(feature_dict, targets):
    """Convert the data to tf.Example and serialize it."""

    def get_values(tensor):
        """Helper function to get values from either RaggedTensor or EagerTensor"""
        if isinstance(tensor, tf.RaggedTensor):
            return tensor.values.numpy().flatten()
        return tensor.numpy().flatten()

    def get_row_lengths(tensor):
        """Helper function to get row lengths from either RaggedTensor or EagerTensor"""
        if isinstance(tensor, tf.RaggedTensor):
            return tensor.row_lengths().numpy()
        return [tensor.shape[0]]

    feature = {}
    # Handle all feature tensors dynamically
    for name, (is_ragged, tensor) in feature_dict.items():
        if is_ragged:
            feature[f"{name}_rowlens"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=get_row_lengths(tensor))
            )
        feature[f"{name}_values"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=get_values(tensor))
        )
    # Add global features and targets
    feature.update(
        {
            "targets": tf.train.Feature(
                int64_list=tf.train.Int64List(value=get_values(targets))
            )
        }
    )
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(dataset, filename):
    """Write the dataset to a TFRecord file."""
    is_ragged = []
    dataset_spec = dataset.element_spec
    for collection_spec in dataset_spec:
        if (len(collection_spec.shape) == 2) and (
            hasattr(collection_spec, "ragged_rank")
        ):
            is_ragged.append(True)
        elif (len(collection_spec.shape) == 1) and not (
            hasattr(collection_spec, "ragged_rank")
        ):
            is_ragged.append(False)
        else:
            raise ValueError(f"Unexpected element specification: {collection_spec}")
    tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(filename, options=tf_record_options) as writer:
        for record in dataset:
            # Create feature dictionary
            feature_dict = {
                f"collection_{col_i}": (is_ragged[col_i], collection)
                for col_i, collection in enumerate(record[:-1])
            }
            targets = record[-1]
            example = serialize_example(feature_dict, targets)
            writer.write(example)


# Convert element_spec to a serializable dictionary
def serialize_spec(spec):
    if isinstance(spec, tf.TensorSpec):
        return {"dtype": spec.dtype.name, "shape": spec.shape.as_list()}
    elif isinstance(spec, tf.RaggedTensorSpec):
        return {
            "dtype": spec.dtype.name,
            "shape": spec.shape.as_list(),
            "ragged_rank": spec.ragged_rank,
        }
    elif isinstance(spec, dict):
        return {key: serialize_spec(value) for key, value in spec.items()}
    elif isinstance(spec, tuple):
        return tuple(serialize_spec(value) for value in spec)
    elif isinstance(spec, list):
        return list(serialize_spec(value) for value in spec)
    else:
        raise ValueError(f"Unsupported spec type: {type(spec)} in {spec}")
