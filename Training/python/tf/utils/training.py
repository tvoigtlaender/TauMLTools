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

def _load_datasets(files, element_spec=None):
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

def compose_datasets(tf_dataset_cfg, n_gpu, input_dataset_cfg):
    if n_gpu < 1:
        global_batch_multiplier = 1
    else:
        global_batch_multiplier = n_gpu

    train_files = glob(tf_dataset_cfg["datasets_location"]["training"])
    val_files = glob(tf_dataset_cfg["datasets_location"]["validation"])
    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError(f'No training or validation datasets found in {tf_dataset_cfg["datasets_location"]}')
    else:
        print(f'Found {len(train_files)} training datasets in {tf_dataset_cfg["datasets_location"]["training"]}')
        print(f'Found {len(val_files)} validation datasets in {tf_dataset_cfg["datasets_location"]["validation"]}')

    # NOTE: it is necessary to overwrite the reader_func of the loader if used in combination with interleave 
    #   as both the load function AND the interleave function attempt to use all available cores otherwise. 
    #   This leads to an exponential creation of threads for machines with many cores,
    element_spec = tf.data.Dataset.load(train_files[0], compression='GZIP').element_spec
    
    if tf_dataset_cfg['combine_via'] == 'sampling': # compose final dataset as sampling from the set of loaded input TF datasets
        # _dataset = tf.data.Dataset.load(p, compression='GZIP')
        _dataset_train = _load_datasets(train_files, element_spec)
        _dataset_val = _load_datasets(val_files, element_spec)
        train_data = tf.data.Dataset.sample_from_datasets(datasets=_dataset_train, seed=1234, stop_on_empty_dataset=False) # True so that the last batches are not purely of one class
        val_data = tf.data.Dataset.sample_from_datasets(datasets=_dataset_val, seed=1234, stop_on_empty_dataset=False)
        
    elif tf_dataset_cfg['combine_via'] == 'interleave': # compose final dataset as consecutive (cycle_length=1) loading of input TF datasets

        # This code only works up to tf 2.10
        # there is no way to use the load function with interleave after that
        cycle_length = 4
        block_length = 1
        train_data_ = tf.data.Dataset.from_tensor_slices(train_files)
        train_data = train_data_.interleave(
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

        val_data_ = tf.data.Dataset.from_tensor_slices(val_files)
        val_data = val_data_.interleave(
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
    else:
        raise ValueError("`combine_via` should be either 'sampling' or 'interleave'")
        
        
    # shuffle/cache
    if tf_dataset_cfg["shuffle_buffer_size"] is not None:
        train_data = train_data.shuffle(tf_dataset_cfg["shuffle_buffer_size"])
    if tf_dataset_cfg["cache"]:
        train_data = train_data.cache()

    # batch/smart batch
    if tf_dataset_cfg['smart_batching_step'] is None:
        train_data = train_data.batch(tf_dataset_cfg["train_batch_size"] * global_batch_multiplier)
        val_data = val_data.batch(tf_dataset_cfg["val_batch_size"] * global_batch_multiplier)
    else:
        # train_data, val_data = _token_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)
        # train_data, val_data = _smart_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)
        train_data, val_data = _smart_batch_V2(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)

    # Add axis to global collection and make it ragged
    if "global" in list(input_dataset_cfg['feature_names'].keys()):
        glob_index = list(input_dataset_cfg['feature_names'].keys()).index("global")
        train_data = train_data.map(lambda *inputs: (*inputs[:glob_index], tf.RaggedTensor.from_tensor(tf.expand_dims(inputs[glob_index],axis=-2)), *inputs[glob_index+1:]))
        val_data = val_data.map(lambda *inputs: (*inputs[:glob_index], tf.RaggedTensor.from_tensor(tf.expand_dims(inputs[glob_index],axis=-2)), *inputs[glob_index+1:]))
    
    if tf_dataset_cfg["scaler"] is not None:
        # Get scaler data from cfg.yaml files
        scaler_data = {}
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
        scaling_means = {}
        scaling_stds = {}
        for collection in input_dataset_cfg['feature_names']:
            scaling_means[collection] = []
            scaling_stds[collection] = []
            for variable in input_dataset_cfg['feature_names'][collection]:
                scaling_means[collection].append(merged_scaler_data[collection][variable]["mean"])
                scaling_stds[collection].append(merged_scaler_data[collection][variable]["std"])

        # Create scalers
        scalers = [
            tf.keras.layers.Normalization(
                axis=-1, 
                mean=scaling_means[collection], 
                variance=scaling_stds[collection]
            ) for collection in input_dataset_cfg['feature_names']
        ]

        # Define function to apply scalers
        @tf.function
        def apply_scaler(*data, num_collections=len(input_dataset_cfg['feature_names'])):
            dat = []
            for i, collection in enumerate(data[:num_collections]):
                flat_dat = scalers[i](collection.values)
                dat.append(tf.RaggedTensor.from_row_splits(values=flat_dat, row_splits=collection.row_splits))
            dat.append(data[num_collections])
            return dat
    
        # Apply scaler to train and validation data
        train_data = train_data.map(apply_scaler)
        val_data = val_data.map(apply_scaler)
        
    # prefetch
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

    # select from stored labels only those classes which are specified in the training cfg 
    class_idx = [input_dataset_cfg['label_columns'].index(f'label_{c}') for c in tf_dataset_cfg["classes"]]
    train_data = train_data.map(lambda *inputs: (inputs[:-1], tf.gather(inputs[-1], indices=class_idx, axis=-1)),
                                num_parallel_calls=tf.data.AUTOTUNE) # assume that labels tensor is yielded last
    val_data = val_data.map(lambda *inputs: (inputs[:-1], tf.gather(inputs[-1], indices=class_idx, axis=-1)),  
                                num_parallel_calls=tf.data.AUTOTUNE) 

    # if tf_dataset_cfg['smart_batching_step'] is not None:
    #     train_data, val_data = _add_weights_by_size(train_data, val_data)
        
    # limit number of threads, otherwise (n_threads=-1) error pops up (tf.__version__ == 2.9.1)
    options = tf.data.Options()
    options.threading.private_threadpool_size = tf_dataset_cfg["n_threads"]
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    return train_data, val_data

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

def create_padding_mask(seq):
    mask = tf.math.reduce_any(tf.math.not_equal(seq, 0), axis=-1) # [batch, seq], 0 -> padding, 1 -> constituent
    return mask

def log_to_mlflow(model, cfg):
    # save model & print summary
    print("\n-> Saving model")
    path_to_hydra_logs = HydraConfig.get().run.dir
    model.save((f'{path_to_hydra_logs}/{cfg["model"]["name"]}.tf'), save_format="tf") # save to hydra logs
    mlflow.log_artifacts(f'{path_to_hydra_logs}/{cfg["model"]["name"]}.tf', 'model') # and also to mlflow artifacts
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

def _smart_batch_V2(train_data, val_data, global_batch_multiplier, tf_dataset_cfg):
    bucket_boundaries = np.arange(
        tf_dataset_cfg['sequence_length_dist_start']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['sequence_length_dist_end']+tf_dataset_cfg['smart_batching_step'],
        tf_dataset_cfg['smart_batching_step']
    )
    train_batch_sizes = [tf_dataset_cfg['train_batch_size'] * global_batch_multiplier] * (len(bucket_boundaries) + 1)
    val_batch_sizes = [tf_dataset_cfg['val_batch_size'] * global_batch_multiplier] * (len(bucket_boundaries) + 1)
    # Bucket the training dataset by sequence length
    bucketed_train_dataset = train_data.bucket_by_sequence_length(
        element_length_fn,
        bucket_boundaries=bucket_boundaries,  
        bucket_batch_sizes=train_batch_sizes, 
        no_padding=True
    ).shuffle(tf_dataset_cfg['shuffle_smart_buffer_size'])
    # Bucket the validation dataset by sequence length
    bucketed_val_dataset = val_data.bucket_by_sequence_length(
        element_length_fn,
        bucket_boundaries=bucket_boundaries,  
        bucket_batch_sizes=val_batch_sizes,
        # bucket_batch_sizes=val_batch_sizes,
        no_padding=True
    ).shuffle(tf_dataset_cfg['shuffle_smart_buffer_size'])
    return bucketed_train_dataset, bucketed_val_dataset
