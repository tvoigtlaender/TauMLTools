import os
import shutil
import yaml
import hydra
import time
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

def pre_import_gpu(cfg):
    if cfg.get("gpu"):
        gpu_config = cfg["gpu"]
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        else:
            if not gpu_config == "all":
                # Use default if all GPUs are requested
                visible_devices = gpu_config
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        print("Trying to use GPUs: {}".format(visible_devices))
        return visible_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Trying to use no GPUs")
        return ""

@hydra.main(config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:
    
    visible_gpu = pre_import_gpu(cfg)
    
    import tensorflow as tf
    # from tensorflow.keras import mixed_precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)

    from models.taco import TacoNet
    from models.transformer import Transformer, CustomSchedule
    from models.particle_net import ParticleNet
    from utils.training import compose_datasets, log_to_mlflow
    import mlflow
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False) 
    # from checkpointer.keras_callback import KerasCheckpointerCallback
    
    
    cpus = tf.config.list_physical_devices('CPU')
    if cpus and cfg.get("cpu"):
        cpu_config = cfg["cpu"]
        threads_per_core = 2
        n_cpus = int(cpu_config["cores"])
        if not "OMP_NUM_THREADS" in os.environ:
            print('"OMP_NUM_THREADS" is not set defaulting to a maximum of 1 CPU core.')
            cpu_max = 4
        else:
            cpu_max = int(os.getenv("OMP_NUM_THREADS"))
            # Overwrite for solo TOpAS
        cpu_max  = 255
        if n_cpus > cpu_max:
            raise Exception("More CPU cores assigned than available.")
        available_threads = n_cpus * threads_per_core
        print("{} CPUs with a total of {} threads are available.".format(os.getenv("OMP_NUM_THREADS"), available_threads))
        intra_threads = cpu_config["intra"]
        inter_threads = cpu_config["inter"]
        if intra_threads + inter_threads > available_threads:
            raise Exception("More threads assigned ({}, {}) than available ({})".format(
                intra_threads, inter_threads, available_threads
            ))
        print("Using {} for intra op parallelism and {} for inter op parallelism.".format(intra_threads, inter_threads))
        # Overwrite for solo TOpAS
        # tf.config.threading.set_intra_op_parallelism_threads(intra_threads) 
        # tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        tf.config.threading.set_inter_op_parallelism_threads(25)
        tf.config.threading.set_intra_op_parallelism_threads(255)
        tf.config.set_soft_device_placement(True)
        tf.config.optimizer.set_jit(True)
    else:
        print("No cpu config set up. Leaving as is.")

    # setup gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_multiply = len(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(len(gpus), "Physical GPUs found")
        if len(gpus) > 1:
            # use_strategy = tf.distribute.MirroredStrategy()
            use_strategy = tf.distribute.MultiWorkerMirroredStrategy()
            #use_strategy =  tf.distribute.experimental.CentralStorageStrategy()
        else:
            use_strategy = tf.distribute.get_strategy()
    else:
        use_strategy = tf.distribute.get_strategy()
        gpu_multiply = 1
        print("No GPUs found")

    # set up mlflow experiment id
    mlflow.set_tracking_uri(f'file://{os.path.abspath(cfg["path_to_mlflow"])}')
    experiment = mlflow.get_experiment_by_name(cfg["experiment_name"])
    if experiment is not None: # fetch existing experiment id
        run_kwargs = {'experiment_id': experiment.experiment_id}
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg["experiment_name"])
        run_kwargs = {'experiment_id': experiment_id}

    # start mlflow run
    with mlflow.start_run(**run_kwargs) as active_run:
        run_id = active_run.info.run_id
        
        # load cfg used for the dataset composition
        with open(cfg["tf_dataset_cfg"]["datasets_location"]["dataset_cfg"], "r") as f:
            print(f'Loading dataset config from {cfg["input_files"]["cfg"]}.yml')
            input_dataset_cfg = yaml.safe_load(f)

        checkpoint_path = 'tmp_checkpoints'
        with use_strategy.scope():
            # load datasets 
            train_data, val_data = compose_datasets(cfg, gpu_multiply, input_dataset_cfg)

            # define model
            feature_name_to_idx = {}
            for feature_type, feature_names in input_dataset_cfg['feature_names'].items():
                feature_name_to_idx[feature_type] = {name: i for i, name in enumerate(feature_names)}
                
            if cfg["model"]["type"] == 'taco_net':
                model = TacoNet(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"])
            elif cfg["model"]["type"] == 'transformer':
                model = Transformer(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"])
            elif cfg["model"]["type"] == 'transformer_dynamic':
                tf_dataset_cfg = cfg["tf_dataset_cfg"]
                max_batch_size = tf_dataset_cfg['tokens_per_batch'] / (tf_dataset_cfg['sequence_length_dist_start']+tf_dataset_cfg['smart_batching_step'])
                class Transformer_dynamic(Transformer):
                    def __init__(self, feature_name_to_idx, encoder_kwargs, decoder_kwargs, max_batch_size):
                        super().__init__(feature_name_to_idx, encoder_kwargs, decoder_kwargs)
                        self.max_batch_size = max_batch_size
                        
                    def train_step(self, data):
                        # Unpack the data. Assumes data is (inputs, labels).
                        x, y = data

                        # Record the operations for automatic differentiation.
                        with tf.GradientTape() as tape:
                            y_pred = self(x, training=True)  # Forward pass
                            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

                        # Compute gradients
                        gradients = tape.gradient(loss, self.trainable_variables)
                        
                        # Get the current batch size
                        batch_size = tf.cast(len(gradients), tf.float32)
                        
                        # Scale the gradients based on the ratio between batch_size and nominal_batch_size
                        scaling_factor = batch_size / self.max_batch_size
                        scaled_gradients = [grad * scaling_factor for grad in gradients]
                        
                        # Apply gradients to update model weights
                        self.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_variables))
                                # Update metrics (includes the metric that tracks the loss)

                        # for metric in self.metrics:
                        #     if metric.name == "loss":
                        #         metric.update_state(loss)
                        #     else:
                        #         metric.update_state(y, y_pred)

                        # Update the metrics.
                        self.compiled_metrics.update_state(y, y_pred)
                        
                        # Return a dictionary of metric results.
                        return {m.name: m.result() for m in self.metrics}
                
                model = Transformer_dynamic(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"], max_batch_size)

            elif cfg['model']['type'] == 'particle_net':
                model = ParticleNet(feature_name_to_idx, cfg['model']['kwargs']['encoder'], cfg['model']['kwargs']['decoder'])
            else:
                raise RuntimeError('Failed to infer model type')
            
            # Currently hardcoded sizes
            input_shapes = [
                (None, None, 35),  # First feature tensor
                (None, None, 74),  # Second feature tensor
                (None, None, 36),  # Third feature tensor 
                (None, 1, 43)      # Global features tensor
            ]
            model.build(input_shapes)

            # LR schedule
            if cfg['schedule'] is None: 
                learning_rate = cfg["learning_rate"]
            elif cfg['schedule']=='custom':
                learning_rate = CustomSchedule(float(cfg["model"]["kwargs"]["encoder"]["dim_model"]), float(cfg['warmup_steps']), float(cfg['lr_multiplier']))
            elif cfg['schedule']=='decrease':
                def scheduler(epoch, lr):
                    if epoch%cfg['decrease_every']!=0 or epoch==0:
                        return lr
                    else:
                        return lr / cfg['decrease_by']
                learning_rate = cfg["learning_rate"]
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
            else:
                raise RuntimeError(f"Unknown value for schedule: {cfg['schedule']}. Only \'custom\', \'decrease\' and \'null\' are supported.")

            # optimiser
            if cfg['optimiser']=='adam': 
                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
            elif cfg['optimiser']=='sgd':
                opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cfg['momentum'], nesterov=cfg['nesterov'])
            elif cfg['optimiser']=='adamw':
                import tensorflow_addons as tfa
                opt = tfa.optimizers.AdamW(weight_decay=cfg['weight_decay'], learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
            elif cfg['optimiser']=='radam':
                import tensorflow_addons as tfa
                opt = tfa.optimizers.RectifiedAdam(weight_decay=cfg['weight_decay'], learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
            else:
                raise RuntimeError(f"Unknown value for optimiser: {cfg['optimiser']}. Only \'sgd\' and \'adam\' are supported.")
            # opt = mixed_precision.LossScaleOptimizer(opt, dynamic=True)

            # callbacks, compile, fit
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=cfg["min_delta"], patience=cfg["patience"], mode='auto', restore_best_weights=True)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path + "/" + "epoch_{epoch:02d}---val_loss_{val_loss:.3f}",
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_freq='epoch',
                save_best_only=False)
            # backup_path = str(os.getcwd() + "/tmp_backup")
            # try:
            #     os.mkdir(backup_path)
            # except:
            #     pass
            # checkpointing = KerasCheckpointerCallback( # setting up the checkpointer callback
            #     local_checkpoint_file=backup_path, # local checkpoint file
            #     checkpoint_every=1, # checkpointing every epoch
            #     checkpoint_transfer_mode="xrootd", # using a shared filesystem to move hte checkpoint to a persisten storage
            #     checkpoint_transfer_target="/store/user/tvoigtlaender/checkpoints/DeepTau/2024_aug_l",
            #     xrootd_server_name="root://cmsdcache-kit-disk.gridka.de/",
            # )


            path_to_hydra_logs = HydraConfig.get().run.dir
            tensorboard_logdir = f'{path_to_hydra_logs}/custom_tensorboard_logs'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, profile_batch = (10, 20))

            callbacks = [early_stopping, model_checkpoint, tensorboard_callback]
            # callbacks = [early_stopping, model_checkpoint] #, checkpointing]
            # callbacks = [early_stopping, model_checkpoint, tensorboard_callback] #, checkpointing]
            if cfg['schedule']=='descrease':
                callbacks.append(lr_scheduler)
            model.compile(optimizer=opt,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                        metrics=['accuracy', tf.keras.metrics.AUC(from_logits=False)])
        start_time = time.time()
        model.fit(train_data, validation_data=val_data, epochs=cfg["n_epochs"], callbacks=callbacks, verbose=1) #, steps_per_epoch=300, validation_steps=300)
        end_time = time.time()
        print("Runtime: {}".format(end_time-start_time))
        # log info
        log_to_mlflow(model, cfg)
        mlflow.log_param('run_id', run_id)
        mlflow.log_artifacts(checkpoint_path, "checkpoints")
        shutil.rmtree(checkpoint_path)

        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg["experiment_name"]}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')

if __name__ == '__main__':
    main()
