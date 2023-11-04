import os
import shutil
import yaml
import hydra
import time
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import tensorflow_addons as tfa

from models.taco import TacoNet
from models.transformer import Transformer, CustomSchedule
from models.particle_net import ParticleNet
from utils.training import compose_datasets, log_to_mlflow

import mlflow
mlflow.tensorflow.autolog(log_models=False) 

@hydra.main(config_path='configs', config_name='train')
def main(cfg: DictConfig) -> None:
    cpus = tf.config.list_physical_devices('CPU')
    if cpus:
        cpu_config = cfg["cpu"]
        threads_per_core = 2
        n_cpus = int(cpu_config["cores"])
        if not "OMP_NUM_THREADS" in os.environ:
          print('"OMP_NUM_THREADS" is not set defaulting to a maximum of 1 CPU core.')
          cpu_max = 1
        else:
          cpu_max = int(os.getenv("OMP_NUM_THREADS"))
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
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    # setup gpu

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_config = cfg["gpu"]
        print("Use GPUs {}.".format(gpu_config['gpu_index']))
        try:
            use_gpus = gpu_config['gpu_index']
            if isinstance(use_gpus, int):
                use_gpus = [use_gpus]
            valid_gpus = [gpu for i_gpu, gpu in enumerate(gpus) if i_gpu in use_gpus]
            tf.config.set_visible_devices(valid_gpus, 'GPU')
            for gpu in valid_gpus:
                if 'gpu_mem' in gpu_config.keys():
                    print("Set device memory limit of {} to {}GB.".format(gpu, gpu_config['gpu_mem']))
                    tf.config.experimental.set_virtual_device_configuration(gpu,
                       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_config['gpu_mem']*1024)])
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
        if len(gpus) > 1:
            use_strategy = tf.distribute.MirroredStrategy()
            #use_strategy = tf.distribute.MultiWorkerMirroredStrategy()
            #use_strategy =  tf.distribute.experimental.CentralStorageStrategy()
        else:
            use_strategy = tf.distribute.get_strategy()
    else:
        use_strategy = tf.distribute.get_strategy()
        logical_gpus = []
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
        with open(cfg['input_dataset_cfg'], "r") as f:
            input_dataset_cfg = yaml.safe_load(f)

        with use_strategy.scope():
            # load datasets 
            train_data, val_data = compose_datasets(cfg["datasets"], cfg["tf_dataset_cfg"], len(logical_gpus), input_dataset_cfg)

            # define model
            feature_name_to_idx = {}
            for feature_type, feature_names in input_dataset_cfg['feature_names'].items():
                feature_name_to_idx[feature_type] = {name: i for i, name in enumerate(feature_names)}
            if cfg["model"]["type"] == 'taco_net':
                model = TacoNet(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"])
            elif cfg["model"]["type"] == 'transformer':
                model = Transformer(feature_name_to_idx, cfg["model"]["kwargs"]["encoder"], cfg["model"]["kwargs"]["decoder"])
            elif cfg['model']['type'] == 'particle_net':
                model = ParticleNet(feature_name_to_idx, cfg['model']['kwargs']['encoder'], cfg['model']['kwargs']['decoder'])
            else:
                raise RuntimeError('Failed to infer model type')
            X_, _ = next(iter(train_data))
            model(X_) # init it for correct autologging with mlflow

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
                opt = tfa.optimizers.AdamW(weight_decay=cfg['weight_decay'], learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
            elif cfg['optimiser']=='radam':
                opt = tfa.optimizers.RectifiedAdam(weight_decay=cfg['weight_decay'], learning_rate=learning_rate, beta_1=cfg['beta_1'], beta_2=cfg['beta_2'], epsilon=cfg['epsilon'])
            else:
                raise RuntimeError(f"Unknown value for optimiser: {cfg['optimiser']}. Only \'sgd\' and \'adam\' are supported.")

            # callbacks, compile, fit
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=cfg["min_delta"], patience=cfg["patience"], mode='auto', restore_best_weights=True)
            checkpoint_path = 'tmp_checkpoints'
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path + "/" + "epoch_{epoch:02d}---val_loss_{val_loss:.3f}",
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_freq='epoch',
                save_best_only=False)

            path_to_hydra_logs = HydraConfig.get().run.dir
            tensorboard_logdir = f'{path_to_hydra_logs}/custom_tensorboard_logs'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, profile_batch = (100, 300))

            callbacks = [early_stopping, model_checkpoint, tensorboard_callback]
            if cfg['schedule']=='descrease':
                callbacks.append(lr_scheduler)
            model.compile(optimizer=opt,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                        metrics=['accuracy', tf.keras.metrics.AUC(from_logits=False)])
        start_time = time.time()
        model.fit(train_data, validation_data=val_data, epochs=cfg["n_epochs"], callbacks=callbacks, verbose=1)  #  steps_per_epoch=1000, 
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
