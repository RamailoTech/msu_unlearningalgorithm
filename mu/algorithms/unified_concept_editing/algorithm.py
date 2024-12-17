# unified_concept_editing/algorithm.py

from core.base_algorithm import BaseAlgorithm
from algorithms.unified_concept_editing.model import UnifiedConceptEditingModel
from algorithms.unified_concept_editing.trainer import UnifiedConceptEditingTrainer
from algorithms.unified_concept_editing.data_handler import UnifiedConceptEditingDataHandler
import torch
import wandb
import logging
from typing import Dict


class UnifiedConceptEditingAlgorithm(BaseAlgorithm):
    """
    UnifiedConceptEditingAlgorithm orchestrates the training process for the Unified Concept Editing method.
    """

    def __init__(self, config: Dict):
        """
        Initialize the UnifiedConceptEditingAlgorithm.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.model = None
        self.trainer = None
        self.data_handler = None
        self.device = torch.device(self.config.get('devices', ['cuda:0'])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up components...")
        
        # Initialize Data Handler
        self.data_handler = UnifiedConceptEditingDataHandler(

            selected_theme=self.config.get('theme'),
            selected_class=self.config.get('classes'),
            use_sample=self.config.get('use_sample', False),

        )

        # Initialize Model
        self.model = UnifiedConceptEditingModel(
            ckpt_path=self.config.get('ckpt_path'),
            device=str(self.device)
        )

        # Initialize Trainer
        self.trainer = UnifiedConceptEditingTrainer(
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
            project='unified-concept-editing',
            name=self.config.get('theme', 'UnifiedConceptEditing'),
            config=self.config
        )
        self.logger.info("Initialized WandB for logging.")

        # Start training
        self.trainer.train()

        # Finish WandB run
        wandb.finish()
