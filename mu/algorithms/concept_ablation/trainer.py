from typing import Dict
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
import logging
from pytorch_lightning import seed_everything
import os 
from packaging import version
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import  LearningRateMonitor
from pytorch_lightning.trainer import Trainer
import signal
import pudb
from pathlib import Path
from argparse import Namespace
import argparse
from stable_diffusion.ldm.util import instantiate_from_config

from mu.algorithms.concept_ablation.data_handler import ConceptAblationDataHandler
from mu.core.base_trainer import BaseTrainer


class ConceptAblationTrainer(BaseTrainer):
    """
    Trainer class for the Concept Ablation algorithm.
    Handles the training loop, loss computation, and optimization.
    """

    def __init__(self, model, config: Dict, device: str, config_path: str, **kwargs):
        """
        Initialize the ConceptAblationTrainer.

        Args:
            model: Instance of ConceptAblationModel or a similar model class.
            config (dict): Configuration dictionary with training parameters.
            device (str): Device to perform training on (e.g. 'cuda:0').
            data_handler: An instance of ConceptAblationDataHandler or a similar data handler.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model, config, **kwargs)
        self.device = device
        self.model = model.model
        self.model_config_path = model.model_config_path
        self.config_path = config_path
        self.opt_config = config
        self.logger = logging.getLogger(__name__)
        self.setup_optimizer()

    def setup_optimizer(self,*args, **kwargs):
        """
        Setup the optimizer based on the training configuration.
        Adjust parameter groups or other attributes as per concept ablation needs.
        """
        pass

    def train(self):
        """
        Execute the training loop.
        """

        configs = [OmegaConf.load(cfg) for cfg in [self.config_path, self.model_config_path]]
        config = OmegaConf.merge(*configs)

        lightning_config = config.pop("lightning", OmegaConf.create())

        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        trainer_config["accelerator"] = "gpu"

        trainer_config["devices"] = self.opt_config.get('devices')

        
        trainer_config["strategy"] = "ddp"
        cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)

        lightning_config.trainer = trainer_config

        if self.opt_config.get('prompts') is None:
            if self.opt_config.get('datapath') == '' or self.opt_config.get('caption') == '':
                self.logger.info('either initial prompts or path to generated images folder should be provided')
                raise NotImplementedError
            if self.opt_config.get('regularization'):
                config.datapath2 = self.opt_config.get('datapath')
                config.caption2 = self.opt_config.get('caption')
        else:
            name = Path(self.opt_config.get('prompts')).stem
            gen_folder = Path(self.opt_config.get('processed_dataset_dir')) / (name + '_gen')
            os.makedirs(gen_folder, exist_ok=True)
            ranks = [int(i) for i in trainer_config["devices"].split(',') if i != ""]
            ConceptAblationDataHandler.preprocess(self.opt_config, self.model_config_path, gen_folder, ranks)
            self.opt_config['datapath'] = str(gen_folder / 'images.txt')
            self.opt_config['caption'] = str(gen_folder / 'caption.txt')
            if self.opt_config.get('regularization'):
                self.opt_config['datapath2'] = str(gen_folder / 'images.txt')
                self.opt_config['caption2'] = str(gen_folder / 'caption.txt')

        config.data.params.train.params.caption = self.opt_config.get('caption')
        config.data.params.train.params.reg_caption = self.opt_config.get('reg_caption')
        config.data.params.train.params.datapath = self.opt_config.get('datapath')
        config.data.params.train.params.reg_datapath = self.opt_config.get('reg_datapath')
        if self.opt_config.get('caption2') is not None:
            config.data.params.train2.params.caption = self.opt_config.get('caption2')
            config.data.params.train2.params.reg_caption = self.opt_config.get('reg_caption2')
            config.data.params.train2.params.datapath = self.opt_config.get('datapath2')
            config.data.params.train2.params.reg_datapath = self.opt_config.get('reg_datapath2')

        trainer_kwargs = dict()

        seed = self.opt_config.get('seed', 42)
        wandb_entity = self.opt_config.get('wandb_entity', "")
        output_dir = self.opt_config.get('output_dir', "")

        os.makedirs(output_dir, exist_ok=True)

        ckptdir = os.path.join(output_dir, "checkpoints")
        cfgdir = os.path.join(output_dir, "configs")

        self.logger.info("Starting training...")
        seed_everything(seed)

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": "quick-canvas-machine-unlearning",
                    "name": f"concept_ablation_{self.opt_config.get('template_name')}",
                    "save_dir": output_dir,
                    "dir": output_dir,
                    "id": f"concept_ablation_{self.opt_config.get('template_name')}",
                    "resume": "allow",
                    "entity": 'concept-ablation',
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": output_dir,
                }
            },
        }
        if wandb_entity is not None:
            default_logger_cfg = default_logger_cfgs["wandb"]
        else:
            default_logger_cfg = default_logger_cfgs["tensorboard"]

        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(self.model, "monitor"):
            self.logger.info(f"Monitoring {self.model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = self.model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = -1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        self.logger.info(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "mu.algorithms.concept_ablation.callbacks.SetupCallback",
                "params": {
                    "resume": "",
                    "now": f"concept_ablation_{self.opt_config.get('template_name')}",
                    "logdir": output_dir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "mu.algorithms.concept_ablation.callbacks.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "mu.algorithms.concept_ablation.trainer.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "mu.algorithms.concept_ablation.callbacks.CUDACallback"
            },
        }

        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            self.logger.info(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': modelckpt_cfg.param.every_n_train_steps,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
        
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if ('ignore_keys_callback' in callbacks_cfg) and self.opt_config.get('ckpt_path'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = self.opt_config.get('ckpt_path')
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt,**trainer_kwargs)
        # trainer = Trainer.from_argparse_args(**trainer_kwargs)
        
        trainer.logdir = output_dir 
        
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()

        self.logger.info("#### Data #####")
        for k in data.datasets:
            self.logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        self.logger.info(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if config.get('scale_lr'):
            self.model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            self.logger.info(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    self.model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            self.model.learning_rate = base_lr
            self.logger.info("++++ NOT USING LR SCALING ++++")
            self.logger.info(f"Setting learning rate to {self.model.learning_rate:.2e}")

        # allow checkpointing via USR1

        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                self.logger.info("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                pudb.set_trace()


        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        try:
            trainer.fit(self.model, data)
        except Exception:
            melk()
            raise


