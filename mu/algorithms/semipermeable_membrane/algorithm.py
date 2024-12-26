# semipermeable_membrane/algorithm.py

import logging
from typing import Dict

import torch
import wandb
from algorithms.semipermeable_membrane.data_handler import (
    SemipermeableMembraneDataHandler,
)
from algorithms.semipermeable_membrane.model import SemipermeableMembraneModel
from algorithms.semipermeable_membrane.trainer import SemipermeableMembraneTrainer


class SemipermeableMembraneAlgorithm:
    """
    SemipermeableMembraneAlgorithm orchestrates the setup and training of the SPM method.
    """

    def __init__(self, config: Dict):
        """
        Initialize the SemipermeableMembraneAlgorithm.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.model = None
        self.trainer = None
        self.data_handler = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up components...")

        # Initialize Data Handler
        self.data_handler = SemipermeableMembraneDataHandler(
            selected_theme=self.config.get('theme', ''),
            selected_class=self.config.get('classes', ''),
            use_sample=self.config.get('use_sample', False)
        )

        # Initialize Model
        self.model = SemipermeableMembraneModel(self.config)

        # Initialize Trainer
        self.trainer = SemipermeableMembraneTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            data_handler=self.data_handler
        )

    def run(self):
        """
        Execute the training process.
        """
        # Initialize WandB
        wandb.init(
            project=self.config.get('wandb', {}).get('project', 'semipermeable_membrane_project'),
            name=self.config.get('wandb', {}).get('name', 'spm_run'),
            config=self.config
        )
        self.logger.info("Initialized WandB for logging.")

        # Start training
        self.trainer.train()

        # Finish WandB run
        wandb.finish()
