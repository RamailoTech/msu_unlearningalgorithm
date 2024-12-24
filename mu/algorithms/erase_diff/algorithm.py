# erase_diff/algorithm.py

from core.base_algorithm import BaseAlgorithm
from algorithms.erase_diff.model import EraseDiffModel
from algorithms.erase_diff.trainer import EraseDiffTrainer
from algorithms.erase_diff.sampler import EraseDiffSampler
from algorithms.erase_diff.data_handler import EraseDiffDataHandler
import torch
import wandb
from typing import Dict
import logging


class EraseDiffAlgorithm(BaseAlgorithm):
    """
    EraseDiffAlgorithm orchestrates the training process for the EraseDiff method.
    """

    def __init__(self, config: Dict):
        """
        Initialize the EraseDiffAlgorithm.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.model = None
        self.trainer = None
        self.sampler = None
        self.data_handler = None
        self.device = torch.device(self.config.get('devices', ['cuda:0'])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """
        Setup model, data handler, trainer, and sampler components.
        """
        self.logger.info("Setting up components...")
        # Initialize Data Handler

        self.data_handler = EraseDiffDataHandler(
            original_data_dir=self.config.get('original_data_dir'),
            new_data_dir=self.config.get('new_data_dir'),
            selected_theme=self.config.get('theme'),
            selected_class=self.config.get('class'), 
            batch_size=self.config.get('batch_size', 4),
            image_size=self.config.get('image_size', 512),
            interpolation=self.config.get('interpolation', 'bicubic'),
            use_sample=self.config.get('use_sample', False),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )

        # Initialize Model
        self.model = EraseDiffModel(
            config_path=self.config.get('config_path'),
            ckpt_path=self.config.get('ckpt_path'),
            device=str(self.device)
        )

        # Initialize Trainer
        self.trainer = EraseDiffTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            data_handler=self.data_handler
        )

        # Initialize Sampler
        self.sampler = EraseDiffSampler(
            model=self.model,
            config=self.config,
            device=str(self.device)
        )

    def run(self):
        """
        Execute the training process.
        """
        # Initialize WandB
        wandb.init(project='quick-canvas-machine-unlearning', name=self.config.get('theme', 'EraseDiff'), config=self.config)
        self.logger.info("Initialized WandB for logging.")

        # Start training
        trained_model = self.trainer.train()

        # Save the trained model
        output_name = self.config.get('output_name', 'erase_diff_model.pth')
        self.model.save_model(output_name)
        self.logger.info(f"Trained model saved at {output_name}")
        wandb.save(output_name)

        # Finish WandB run
        wandb.finish()
