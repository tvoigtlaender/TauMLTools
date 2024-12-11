from glob import glob
from collections import defaultdict
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
import numpy as np
import mlflow
import pickle
import time
import os
import json

def compose_datasets(datasets, tf_dataset_cfg, n_gpu, input_dataset_cfg):
    if n_gpu < 1:
        global_batch_multiplier = 1
    else:
        global_batch_multiplier = n_gpu

    # NOTE: it is necessary to overwrite the reader_func of the loader if used in combination with interleave 
    #   as both the load function AND the interleave function attempt to use all available cores otherwise. 
    #   This leads to an exponential creation of threads for machines with many cores,
    if tf_dataset_cfg['combine_via'] == 'sampling': # compose final dataset as sampling from the set of loaded input TF datasets
        datasets_for_training, sample_probas = _combine_datasets(datasets, load=True), None
        train_data = tf.data.Dataset.sample_from_datasets(datasets=datasets_for_training['train'], weights=sample_probas, seed=1234, stop_on_empty_dataset=False) # True so that the last batches are not purely of one class
        val_data = tf.data.Dataset.sample_from_datasets(datasets=datasets_for_training['val'], seed=1234, stop_on_empty_dataset=False)
    elif tf_dataset_cfg['combine_via'] == 'interleave': # compose final dataset as consecutive (cycle_length=1) loading of input TF datasets
        datasets_for_training = _combine_datasets(datasets, load=False)
        element_spec = tf.data.Dataset.load(
            datasets_for_training['train'][0], 
            # compression='GZIP',
            reader_func=lambda dataset: dataset.interleave(
                lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
            ).element_spec
        cycle_length = 4 
        block_length = 1

        train_data = tf.data.Dataset.from_tensor_slices(datasets_for_training['train'])
        train_data = train_data.interleave(
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
            block_length=block_length,
        )

        val_data = tf.data.Dataset.from_tensor_slices(datasets_for_training['val'])
        val_data = val_data.interleave(
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
            block_length=block_length,
        )
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
        train_data, val_data = _smart_batch(train_data, val_data, global_batch_multiplier, tf_dataset_cfg)

    # Add axis to global collection to 
    if "global" in list(input_dataset_cfg['feature_names'].keys()):
        glob_index = list(input_dataset_cfg['feature_names'].keys()).index("global")
        train_data = train_data.map(lambda *inputs: (*inputs[:glob_index], tf.RaggedTensor.from_tensor(tf.expand_dims(inputs[glob_index],axis=-2)), *inputs[glob_index+1:]))
        val_data = val_data.map(lambda *inputs: (*inputs[:glob_index], tf.RaggedTensor.from_tensor(tf.expand_dims(inputs[glob_index],axis=-2)), *inputs[glob_index+1:]))
        
    # Define function to get scaling from dataset
    def get_train_dat_scaler_batch(input_data, num_collections):
        # Innitialize scaler for each collection
        scalers = [StandardScaler() for i in range(num_collections)]
        # Go through all batches
        for batch in input_data:
            # Go through all particle collections
            for i, collection in enumerate(batch[:num_collections]):
                if collection.values.shape[0] > 0:
                    scalers[i].partial_fit(collection.values)
        scaling_data = []
        # Return data of scalers
        for scaler in scalers:
            scaling_data.append({"mean":scaler.mean_, "var": scaler.var_,})
        return scaling_data
    
    if tf_dataset_cfg["scaler"] is not None:
        if os.path.isfile(tf_dataset_cfg["scaler"]):
            with open(tf_dataset_cfg["scaler"]) as f:
                scaling_data = json.load(f)
            print(f"Scaler loaded from {tf_dataset_cfg['scaler']}.")
        else:
            print(f"Scaler file not found in {tf_dataset_cfg['scaler']}, calculating new scaler.")
            start_time_ = time.time()
            scaling_data = get_train_dat_scaler_batch(train_data, len(input_dataset_cfg['feature_names']))
            end_time_ = time.time()
            print("Time to create scaler: {}".format(end_time_-start_time_))
            # Custom function to serialize NumPy arrays
            def serialize_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # Convert NumPy array to list
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            try:
                os.mkdir(os.path.dirname(tf_dataset_cfg["scaler"]))
            except:
                pass
            with open(tf_dataset_cfg["scaler"], "w") as file:
                json.dump(scaling_data, file, default=serialize_numpy)
            print(f"Scaler saved to {tf_dataset_cfg['scaler']}")
        mlflow.log_artifacts(os.path.dirname(tf_dataset_cfg["scaler"]), "scalers")
                
        scalers = [
            tf.keras.layers.Normalization(
                axis=-1, 
                mean=scaling_data[i]["mean"], 
                variance=scaling_data[i]["var"]
            ) for i in range(len(input_dataset_cfg['feature_names']))
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

def _combine_datasets(datasets, load=False):
    datasets_for_training = {'train': [], 'val': []} # to accumulate final datasets
    for dataset_type in datasets_for_training.keys():
        if dataset_type not in datasets:
            raise RuntimeError(f'key ({dataset_type}) should be present in dataset yaml configuration')
        for dataset_name, path_to_dataset in datasets[dataset_type].items(): # loop over specified train/val datasets
            for p in glob(
                '{}/{}/{}/*/'.format(
                    path_to_dataset,
                    dataset_name,
                    dataset_type,
                )
            ): # loop over all globbed files in the dataset
                if load:
                    _dataset = tf.data.Dataset.load(
                        p, 
                        compression='GZIP',
                        reader_func=lambda dataset: dataset.interleave(
                            lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
                        )
                    # _dataset = tf.data.Dataset.load(p, compression='GZIP')
                    datasets_for_training[dataset_type].append(_dataset) 
                else:   
                    datasets_for_training[dataset_type].append(p)
        # datasets_for_training[dataset_type] = datasets_for_training[dataset_type][:5]
    return datasets_for_training

def _combine_for_sampling(datasets):
    # NB: this is a deprecated function
    # keeping it as an example of uniform sampling across training classes
    sample_probas = [] # to store sampling probabilites on training datasets
    datasets_for_training = {'train': [], 'val': []} # to accumulate final datasets
    for dataset_type in datasets_for_training.keys():
        ds_per_tau_type = defaultdict(list)
        if dataset_type not in datasets:
            raise RuntimeError(f'key ({dataset_type}) should be present in dataset yaml configuration')
        for dataset_name, dataset_cfg in datasets[dataset_type].items(): # loop over specified train/val datasets
            for tau_type in dataset_cfg["tau_types"]: # loop over tau types specified for this dataset
                for p in glob(f'{dataset_cfg["path_to_dataset"]}/{dataset_name}/{dataset_type}/*/{tau_type}'): # loop over all globbed files in the dataset
                    dataset = tf.data.load(p) 
                    ds_per_tau_type[tau_type].append(dataset) # add TF dataset (1 input file, 1 tau type) to the map  
        
        n_tau_types = len(ds_per_tau_type.keys())
        for tau_type, ds_list in ds_per_tau_type.items():
            datasets_for_training[dataset_type] += ds_list # add datasets to the final list
            if dataset_type == "train": # for training dataset also keep corresponding sampling probas over input files
                n_files = len(ds_list)
                sample_probas += n_files*[1./(n_tau_types*n_files) ]
    
    return datasets_for_training, sample_probas

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
    mlflow.log_param('datasets_train', cfg["datasets"]["train"].keys())
    mlflow.log_param('datasets_val', cfg["datasets"]["val"].keys())
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
