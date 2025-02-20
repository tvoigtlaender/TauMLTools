from glob import glob
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
import numpy as np
import mlflow
import os
import yaml

def compose_datasets(cfg, n_gpu, input_dataset_cfg):
    tf_dataset_cfg=cfg["tf_dataset_cfg"]
    # Load data from files
    train_data, val_data = load_data(cfg, input_dataset_cfg)
        
    # shuffle/cache
    if tf_dataset_cfg["shuffle_buffer_size"] is not None:
        train_data = train_data.shuffle(tf_dataset_cfg["shuffle_buffer_size"])
    if tf_dataset_cfg["cache"]:
        train_data = train_data.cache()

    # batch/smart batch
    if n_gpu < 1:
        global_batch_multiplier = 1
    else:
        global_batch_multiplier = n_gpu
        
    if tf_dataset_cfg['batching'] == "standard":
        train_data = train_data.batch(tf_dataset_cfg["train_batch_size"] * global_batch_multiplier)
        val_data = val_data.batch(tf_dataset_cfg["val_batch_size"] * global_batch_multiplier)
    elif tf_dataset_cfg['batching'] == "smart":
        train_data, val_data = _smart_batch_V2(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)
        # train_data, val_data = _smart_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)
    elif tf_dataset_cfg['batching'] == "token":
        train_data, val_data = _token_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)
        # if tf_dataset_cfg['smart_batching_step'] is not None:
        #     train_data, val_data = _add_weights_by_size(train_data, val_data)
    else:
        raise ValueError(f"Unsupported batching method: {tf_dataset_cfg['batching']}")

    # Add axis to global collection and make it ragged
    if "global" in list(input_dataset_cfg['feature_names'].keys()):
        glob_index = list(input_dataset_cfg['feature_names'].keys()).index("global")
        train_data = train_data.map(lambda *inputs: (*inputs[:glob_index], tf.expand_dims(inputs[glob_index],axis=-2), *inputs[glob_index+1:]))
        val_data = val_data.map(lambda *inputs: (*inputs[:glob_index], tf.expand_dims(inputs[glob_index],axis=-2), *inputs[glob_index+1:]))

    if tf_dataset_cfg["scaler"]:
        train_data, val_data = scale_data(train_data, val_data, cfg, input_dataset_cfg)
        
    @tf.function
    def ragged_to_dense_and_select_classes(*inputs, class_idx):
        # First convert ragged tensors to dense
        dense_tensors = tuple(input_tensor.to_tensor() if isinstance(input_tensor, tf.RaggedTensor) else input_tensor
                            for input_tensor in inputs)
        # Split features and labels
        features = dense_tensors[:-1]  # All tensors except the last one
        labels = tf.gather(dense_tensors[-1], indices=class_idx, axis=-1)  # Get selected classes from last tensor
        return (features, labels)  # Return as tuple of (features_tuple, labels)
        
        
        # Apply the combined mapping function to the dataset
    class_idx = [input_dataset_cfg['label_columns'].index(f'label_{c}') for c in tf_dataset_cfg["classes"]]
    train_data = train_data.map(
        lambda *inputs: ragged_to_dense_and_select_classes(*inputs, class_idx=class_idx),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_data = val_data.map(
        lambda *inputs: ragged_to_dense_and_select_classes(*inputs, class_idx=class_idx),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # prefetch
    train_data = train_data.prefetch(n_gpu*2)
    val_data = val_data.prefetch(n_gpu*2)
    # train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    # val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # options.threading.private_threadpool_size = tf_dataset_cfg["n_threads"]
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    return train_data, val_data

# def create_padding_mask(seq):
#     mask = tf.math.reduce_any(tf.math.not_equal(seq, 0), axis=-1) # [batch, seq], 0 -> padding, 1 -> constituent
#     return mask

def log_to_mlflow(model, cfg):
    # save model & print summary
    print("\n-> Saving model")
    path_to_hydra_logs = HydraConfig.get().run.dir
    model.save((f'{path_to_hydra_logs}/{cfg["model"]["name"]}'), save_format="tf") # save to hydra logs
    mlflow.log_artifacts(f'{path_to_hydra_logs}/{cfg["model"]["name"]}', 'model') # and also to mlflow artifacts
    if cfg["model"]["type"] == 'taco_net':
        print(model.wave_encoder.summary())
        summary_list_encoder, summary_list_decoder = [], []
        model.wave_encoder.summary(print_fn=summary_list_encoder.append)
        summary_encoder, summary_decoder = "\n".join(summary_list_encoder), "\n".join(summary_list_decoder)
        mlflow.log_text(summary_encoder, artifact_file="encoder_summary.txt")
    elif cfg["model"]["type"] == 'transformer':
        print(model.summary())
    elif cfg['model']['type'] == 'particle_net':
        print(model.summary())

    # log data params
    mlflow.log_param('dataset_name', cfg["dataset_name"])
    mlflow.log_param('datasets_cfg',  cfg["input_files"]["cfg"])
    mlflow.log_param('datasets_train',  cfg["input_files"]["train"])
    mlflow.log_param('datasets_val',  cfg["input_files"]["val"])
    mlflow.log_params(cfg['tf_dataset_cfg'])

    # log model params
    params_encoder = OmegaConf.to_object(cfg["model"]["kwargs"]["encoder"])
    params_embedding = params_encoder.pop('embedding_kwargs')
    params_embedding = {f'emb_{p}': v for p,v in params_embedding.items()}
    mlflow.log_param('model_name', cfg["model"]["name"])
    mlflow.log_params(params_encoder)
    for ptype, feature_list in params_embedding['emb_features_to_drop'].items():
        if len(feature_list)>5:
            params_embedding['emb_features_to_drop'][ptype] = ['too_long_to_log']
    mlflow.log_params(params_embedding)
    mlflow.log_params(cfg["model"]["kwargs"]["decoder"])
    mlflow.log_params({f'model_node_{i}': c for i,c in enumerate(cfg["tf_dataset_cfg"]["classes"])})
    if cfg['schedule']=='decrease':
        mlflow.log_param('decrease_every', cfg['decrease_every'])
        mlflow.log_param('decrease_by', cfg['decrease_by'])
    
    # log N trainable params 
    summary_list = []
    model.summary(print_fn=summary_list.append)
    for l in summary_list:
        if (s:='Trainable params: ') in l:
            mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

# Currently hard-coded for 4 collections with one global one
def element_length_fn(*seq):
    # Sum of tokens in non-global + 1
    return tf.reduce_sum([tf.shape(seq[i])[0] for i in range(3)]) + 1

def _token_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg):
    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['sequence_length_dist_end']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['smart_batching_step']
    )
    # train_batch_sizes = (tf_dataset_cfg['train_tokens_per_batch']/bucket_boundaries).astype(int)
    # val_batch_sizes = (tf_dataset_cfg['val_tokens_per_batch']/bucket_boundaries).astype(int)
    batch_sizes = (tf_dataset_cfg['tokens_per_batch']/bucket_boundaries).astype(int)
    # train_batch_sizes = np.append(train_batch_sizes, int(train_batch_sizes[-1]/2)) * global_batch_multiplier
    # val_batch_sizes = np.append(val_batch_sizes, int(val_batch_sizes[-1]/2)) * global_batch_multiplier
    batch_sizes = np.append(batch_sizes, int(batch_sizes[-1]/2)) * global_batch_multiplier
    # Bucket the dataset by sequence length
    bucketed_train_dataset = train_data.bucket_by_sequence_length(
        element_length_fn,
        bucket_boundaries=bucket_boundaries,  
        bucket_batch_sizes=batch_sizes, 
        # bucket_batch_sizes=train_batch_sizes,
        no_padding=True
    ).shuffle(tf_dataset_cfg['shuffle_smart_buffer_size'])
    # Bucket the dataset by sequence length
    bucketed_val_dataset = val_data.bucket_by_sequence_length(
        element_length_fn,
        bucket_boundaries=bucket_boundaries,  
        bucket_batch_sizes=batch_sizes,
        # bucket_batch_sizes=val_batch_sizes,
        no_padding=True
    ).shuffle(tf_dataset_cfg['shuffle_smart_buffer_size'])
    return bucketed_train_dataset, bucketed_val_dataset

def _smart_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg):
    # will do smart batching based only on the sequence lengths of the **first** element (assume it to be PF candidate block)
    # NB: careful when dropping whole blocks in `embedding.yaml` -> change smart batching id here accordingly
    element_length_func = lambda *elements: tf.shape(elements[0])[0]

    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start'],
        tf_dataset_cfg['sequence_length_dist_end'],
        tf_dataset_cfg['smart_batching_step']
    )
    
    def _element_to_bucket_id(*args):
        seq_length = element_length_func(*args)
        boundaries = list(bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = math_ops.logical_and(
            math_ops.less_equal(buckets_min, seq_length),
            math_ops.less(seq_length, buckets_max)
        )
        bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))
        return bucket_id
    
    def _reduce_func(unused_arg, dataset, batch_size):
        return dataset.batch(batch_size)

    train_data = train_data.group_by_window(
        key_func=_element_to_bucket_id,
        reduce_func=lambda unused_arg, dataset: _reduce_func(unused_arg, dataset, tf_dataset_cfg['train_batch_size'] * global_batch_multiplier),
        window_size=tf_dataset_cfg['train_batch_size'] * global_batch_multiplier
    ).shuffle(int(tf_dataset_cfg['shuffle_smart_buffer_size'] / global_batch_multiplier))

    val_data = val_data.group_by_window(
        key_func=_element_to_bucket_id,
        reduce_func=lambda unused_arg, dataset: _reduce_func(unused_arg, dataset, tf_dataset_cfg['val_batch_size'] * global_batch_multiplier),
        window_size=tf_dataset_cfg['val_batch_size'] * global_batch_multiplier
    ).shuffle(int(tf_dataset_cfg['shuffle_smart_buffer_size'] / global_batch_multiplier))

    return train_data, val_data

def _smart_batch_V2(train_data, val_data, global_batch_multiplier, tf_dataset_cfg):
    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['sequence_length_dist_end']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['smart_batching_step']
    )
    train_batch_sizes = [tf_dataset_cfg['train_batch_size'] * global_batch_multiplier] * (len(bucket_boundaries) + 1)
    val_batch_sizes = [tf_dataset_cfg['val_batch_size'] * global_batch_multiplier] * (len(bucket_boundaries) + 1)

    @tf.function #(jit_compile=True)
    def apply_bucketing(dataset, batch_sizes, is_training=True):
        bucketed_dataset = dataset.bucket_by_sequence_length(
            element_length_fn,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=batch_sizes,
            no_padding=True,
            drop_remainder=is_training  # Drop incomplete batches only during training
        )
        # Only shuffle for training
        if is_training:
            shuffle_size = tf_dataset_cfg['shuffle_smart_buffer_size']
            bucketed_dataset = bucketed_dataset.shuffle(shuffle_size)
        return bucketed_dataset

    # Apply bucketing with prefetch
    train_dataset = apply_bucketing(train_data, train_batch_sizes, True)
    val_dataset = apply_bucketing(val_data, val_batch_sizes, False)
    
    return train_dataset, val_dataset

def load_data(cfg, input_dataset_cfg):
    tf_dataset_cfg=cfg["tf_dataset_cfg"]    
    train_files = [os.path.join(tf_dataset_cfg["datasets_location"]["training"], os.path.basename(f)) for f in cfg["input_files"]["train"]]
    val_files = [os.path.join(tf_dataset_cfg["datasets_location"]["validation"], os.path.basename(f)) for f in cfg["input_files"]["val"]]
    
    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError(f'No training or validation datasets found in {tf_dataset_cfg["datasets_location"]}')
    else:
        print(f'Found {len(train_files)} training datasets in {tf_dataset_cfg["datasets_location"]["training"]}')
        print(f'Found {len(val_files)} validation datasets in {tf_dataset_cfg["datasets_location"]["validation"]}')
        
    # NOTE: it is necessary to overwrite the reader_func of the loader if used in combination with interleave 
    #   as both the load function AND the interleave function attempt to use all available cores otherwise. 
    #   This leads to an exponential creation of threads for machines with many cores,
    if input_dataset_cfg.get("file_format") == "tfrecord":
        element_spec = input_dataset_cfg.get("element_spec")
    elif input_dataset_cfg.get("file_format") == "tfsave":
        element_spec = tf.data.Dataset.load(train_files[0], compression='GZIP').element_spec
    else:
        raise ValueError(f"Unsupported file format: {input_dataset_cfg.get('file_format')}")
    
    if tf_dataset_cfg['combine_via'] == 'sampling': 
        if input_dataset_cfg.get("file_format") == "tfrecord":
            file_names_train = [f"{f}/data.tfrecord" for f in train_files]
            file_names_val = [f"{f}/data.tfrecord" for f in train_files]
            _dataset_train = _read_tfrecord(file_names_train, element_spec)
            _dataset_val = _read_tfrecord(file_names_val, element_spec)
        elif input_dataset_cfg.get("file_format") == "tfsave":
            _dataset_train = _tfsave_load_datasets(train_files, element_spec)
            _dataset_val = _tfsave_load_datasets(val_files, element_spec)
        # True so that the last batches are not purely of one class
        train_data = tf.data.Dataset.sample_from_datasets(datasets=_dataset_train, seed=1234, stop_on_empty_dataset=False) 
        val_data = tf.data.Dataset.sample_from_datasets(datasets=_dataset_val, seed=1234, stop_on_empty_dataset=False)
        
    elif tf_dataset_cfg['combine_via'] == 'interleave': # compose final dataset as consecutive (cycle_length=1) loading of input TF datasets
        if input_dataset_cfg.get("file_format") == "tfrecord":
            file_names_train = [f"{f}/data.tfrecord" for f in train_files]
            file_names_val = [f"{f}/data.tfrecord" for f in train_files]
            train_data = _read_tfrecord(file_names_train, element_spec)
            val_data = _read_tfrecord(file_names_val, element_spec)
        elif input_dataset_cfg.get("file_format") == "tfsave":
            train_data = _tfsave_interleave(train_files, element_spec)
            val_data = _tfsave_interleave(val_files, element_spec)
    else:
        raise ValueError("`combine_via` should be either 'sampling' or 'interleave'")
    
    return train_data, val_data

def scale_data(train_data, val_data, cfg, input_dataset_cfg):
    tf_dataset_cfg=cfg["tf_dataset_cfg"]    
    scaler_data = {}
    train_files = [os.path.join(tf_dataset_cfg["datasets_location"]["training"], os.path.basename(f)) for f in cfg["input_files"]["train"]]
    for file in train_files:
        key = os.path.basename(file)
        with open(f"{file}/cfg.yaml", 'rb') as f:
            scaler_data[key] = yaml.safe_load(f)["scaling_data"]

    # Merge statistics
    merged_scaler_data = merge_statistics(scaler_data, input_dataset_cfg['feature_names'])
    
    # Special treatment for "particle_type" feature as it is categorical and shared between collections
    for collection in input_dataset_cfg['feature_names']:
        if "particle_type" in input_dataset_cfg['feature_names'][collection]:
            merged_scaler_data[collection]["particle_type"]["mean"] = 0.
            merged_scaler_data[collection]["particle_type"]["std"] = 1.

    # Turn merged statistics into lists for Normalization layer
    scaling_means = []
    scaling_stds = []
    for i_col, collection in enumerate(input_dataset_cfg['feature_names']):
        scaling_means.append([])
        scaling_stds.append([])
        for variable in input_dataset_cfg['feature_names'][collection]:
            scaling_means[i_col].append(merged_scaler_data[collection][variable]["mean"])
            scaling_stds[i_col].append(merged_scaler_data[collection][variable]["std"])

    # Convert lists to tf.constant with proper shape and dtype
    scaling_means = [tf.reshape(tf.constant(mean, dtype=tf.float32), [1, 1, -1]) for mean in scaling_means]
    scaling_stds = [tf.reshape(tf.constant(std, dtype=tf.float32), [1, 1, -1]) for std in scaling_stds]

    @tf.function #(jit_compile=True)
    def scale_tensors(*tensors):
        # Process each feature tensor separately
        scaled_features = []
        for feature, mean, std in zip(tensors[:-1], scaling_means, scaling_stds):
            # Scale the feature tensor
            scaled_features.append((feature - mean) / std)
        
        # Return scaled features and original labels
        return tuple(scaled_features) + (tensors[-1],)
            
    # Apply scaling with multiple parallel calls
    train_data = train_data.map(
        scale_tensors,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_data = val_data.map(
        scale_tensors,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_data, val_data

def _tfsave_load_datasets(files, element_spec=None):
    _dataset = []
    for p in files:
        _dataset.append(
            tf.data.Dataset.load(
                p, 
                compression='GZIP',
                reader_func=lambda dataset: dataset.interleave(
                    lambda x: x, 
                    cycle_length=1, 
                    num_parallel_calls=tf.data.AUTOTUNE
                    ),
                element_spec=element_spec
                )
            )
    return _dataset

# This code only works up to tf 2.10
# there is no way to use the load function with interleave after that
def _tfsave_interleave(files, val_files, element_spec):
    cycle_length = 40
    block_length = 1
    data_ds = tf.data.Dataset.from_tensor_slices(files)
    loaded_data = data_ds.interleave(
        lambda x: tf.data.Dataset.load(
            x, 
            element_spec=element_spec, 
            compression='GZIP',
            reader_func=lambda dataset: dataset.interleave(
                lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
            ),
        cycle_length=cycle_length, 
        num_parallel_calls=tf.data.AUTOTUNE, 
        deterministic=False, 
        block_length=block_length)
    return loaded_data

def _load_datasets2(files, element_spec=None):
    return [tf.data.Dataset.load(p, compression='GZIP', element_spec=element_spec,reader_func=lambda dataset: dataset.interleave(lambda x: x, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)) for p in files]

def _merge_datasets2(datasets):
    train_data_ = tf.data.Dataset.from_tensor_slices(datasets)
    train_data = train_data_.interleave(lambda x: x,cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return train_data

def merge_statistics(scaler_data, feature_names):
    """Merge multiple statistics dictionaries for standard scaling.
    Args:
        stats_list (list): A list of dictionaries containing statistics.
    Returns:
        dict: A merged dictionary with global mean, std, min, max, and count.
    """
    merged_stats = {}
    for collection in feature_names:
        merged_stats[collection] = {}
        for variable in feature_names[collection]:
            merged_stats[collection][variable] = {
                    "mean": 0.0, "std": 0.0, "min": float("inf"), "max": float("-inf"), "count": 0
                }
            for file_stats in scaler_data.values():
                # Get existing values
                values = file_stats[collection][variable]
                count_old = merged_stats[collection][variable]["count"]
                count_new = values["count"]
                total_count = count_old + count_new
                # Compute new mean using weighted average
                mean_old = merged_stats[collection][variable]["mean"]
                mean_new = values["mean"]
                merged_mean = (mean_old * count_old + mean_new * count_new) / total_count if total_count > 0 else 0.0
                # Compute new std using pooled variance formula
                std_old = merged_stats[collection][variable]["std"]
                std_new = values["std"]
                merged_variance = (
                    (count_old * (std_old ** 2 + mean_old ** 2) + count_new * (std_new ** 2 + mean_new ** 2))
                    / total_count
                ) - merged_mean ** 2
                merged_std = np.sqrt(max(merged_variance, 0))  # Ensure non-negative variance
                # Update statistics
                merged_stats[collection][variable]["mean"] = merged_mean
                merged_stats[collection][variable]["std"] = merged_std
                merged_stats[collection][variable]["min"] = min(merged_stats[collection][variable]["min"], values["min"])
                merged_stats[collection][variable]["max"] = max(merged_stats[collection][variable]["max"], values["max"])
                merged_stats[collection][variable]["count"] = total_count
    return merged_stats

@tf.function
def _parse_tfrecord(example_proto, dataset_spec):
    """Parse the TFRecord example."""
    feature_description = {
        'targets': tf.io.VarLenFeature(tf.int64)
    }
    is_ragged = []
    dimensions = []
    for col_i, collection_spec in enumerate(dataset_spec[:-1]):
        dimensions.append(collection_spec["shape"][-1])
        if (len(collection_spec["shape"]) == 2) and not (collection_spec.get("ragged_rank") == None):
            is_ragged.append(True)
            feature_description.update({
                f'collection_{col_i}_rowlens': tf.io.VarLenFeature(tf.int64),
            })
        elif (len(collection_spec["shape"]) == 1) and (collection_spec.get("ragged_rank") == None):
            is_ragged.append(False)
        else:
            raise ValueError(f"Unexpected element specification: {collection_spec}")
        feature_description.update({
            f'collection_{col_i}_values': tf.io.VarLenFeature(tf.float32),
        })
    # Add global features and targets
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    feature_tensors = []
    for col_i, (dim, is_ragged) in enumerate(zip(dimensions, is_ragged)):
        if is_ragged:
            values = parsed[f'collection_{col_i}_values'].values
            rowlens = tf.cast(parsed[f'collection_{col_i}_rowlens'].values, tf.int64)
            tensor = tf.RaggedTensor.from_tensor(tf.squeeze(tf.RaggedTensor.from_row_lengths(tf.reshape(values, [-1, dim]),rowlens), axis=0))
            feature_tensors.append(tensor)
        else:
            feature_tensors.append(tf.sparse.to_dense(parsed[f'collection_{col_i}_values']))
    targets = tf.sparse.to_dense(parsed['targets'])
    return tuple(feature_tensors + [targets])

@tf.function
def _read_tfrecord(filenames, element_spec):
    """Read from a TFRecord file and return a dataset."""
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames).shuffle(len(filenames))
    interleaved_dataset = filenames_ds.interleave(
        lambda filenames: tf.data.TFRecordDataset(
            filenames, 
            compression_type='GZIP', 
            num_parallel_reads=tf.data.AUTOTUNE
        ), 
        cycle_length=tf.data.AUTOTUNE,  # Number of parallel file reads
        num_parallel_calls=tf.data.AUTOTUNE
    )
    parsed_dataset = interleaved_dataset.map(
        lambda x: _parse_tfrecord(x, element_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return parsed_dataset
