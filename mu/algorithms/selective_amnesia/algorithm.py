import logging
from typing import Dict
import torch
import wandb
from core.base_algorithm import BaseAlgorithm
from algorithms.selective_amnesia.model import SelectiveAmnesiaModel
from algorithms.selective_amnesia.trainer import SelectiveAmnesiaTrainer
from algorithms.selective_amnesia.data_handler import SelectiveAmnesiaDataHandler

logger = logging.getLogger(__name__)

class SelectiveAmnesiaAlgorithm(BaseAlgorithm):
    """
    Orchestrates the Selective Amnesia training process.
    Sets up model, data handler, and trainer, then runs training.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(self.config.get('devices', ['cuda:0'])[0])
        self.model = None
        self.data_handler = None
        self.trainer = None
        self._setup_components()

    def _setup_components(self):
        logger.info("Setting up Selective Amnesia components...")

        # Data handler: loads q(x|c_f) dataset
        self.data_handler = SelectiveAmnesiaDataHandler(
            surrogate_data_dir=self.config.get('surrogate_data_dir'),
            image_size=self.config.get('image_size', 512),
            interpolation=self.config.get('interpolation', 'bicubic'),
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )

        # Model: Load SD model and FIM
        self.model = SelectiveAmnesiaModel(
            config_path=self.config.get('config_path'),
            ckpt_path=self.config.get('ckpt_path'),
            fim_path=self.config.get('fim_path'),  # path to full_fisher_dict.pkl
            device=str(self.device)
        )

        # Trainer
        self.trainer = SelectiveAmnesiaTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            data_handler=self.data_handler
        )

    def run(self):
        wandb.init(project=self.config.get('project_name', 'selective_amnesia'),
                   name=self.config.get('run_name', 'selective_amnesia_run'),
                   config=self.config)
        logger.info("WandB run started.")
        trained_model = self.trainer.train()
        wandb.finish()
        return trained_model
