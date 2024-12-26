import logging
from typing import Dict

import torch
import wandb
from algorithms.concept_ablation.data_handler import ConceptAblationDataHandler
from algorithms.concept_ablation.model import ConceptAblationModel
from algorithms.concept_ablation.trainer import ConceptAblationTrainer
from core.base_algorithm import BaseAlgorithm


class ConceptAblationAlgorithm(BaseAlgorithm):
    """
    ConceptAblationAlgorithm orchestrates the training process for the Concept Ablation method.
    It sets up the model, data handler, and trainer, and then runs the training loop.
    """

    def __init__(self, config: Dict):
        """
        Initialize the ConceptAblationAlgorithm.

        Args:
            config (Dict): Configuration dictionary containing keys like:
                - 'concept_type': (str) 'style', 'object', or 'memorization'
                - 'prompts_path': (str) path to initial prompts
                - 'output_dir': (str) directory to store generated data and results
                - 'base_config': (str) path to the model config
                - 'ckpt_path': (str) path to the model checkpoint
                - 'delta_ckpt': (str, optional)
                - 'caption_target': (str, optional)
                - 'train_size': (int, optional)
                - 'n_samples': (int, optional)
                - 'image_size': (int, optional)
                - 'interpolation': (str)
                - 'batch_size': (int)
                - 'num_workers': (int)
                - 'pin_memory': (bool)
                - 'use_regularization': (bool, optional)
                - 'devices': (list) CUDA devices to train on
                - 'epochs', 'lr', 'train_method', etc.
                - 'output_name': (str) name of the final saved model file
                - Additional keys for WandB logging, etc.
        """
        self.config = config
        self.device = torch.device(self.config.get('devices', ['cuda:0'])[0])
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.data_handler = None
        self.trainer = None
        self._setup_components()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up Concept Ablation components...")

        # Initialize Data Handler
        self.data_handler = ConceptAblationDataHandler(
            concept_type=self.config.get('concept_type'),
            prompts_path=self.config.get('prompts_path'),
            output_dir=self.config.get('output_dir'),
            base_config=self.config.get('config_path'),
            resume_ckpt=self.config.get('ckpt_path'),
            delta_ckpt=self.config.get('delta_ckpt', None),
            caption_target=self.config.get('caption_target', None),
            train_size=self.config.get('train_size', 1000),
            n_samples=self.config.get('n_samples', 10),
            image_size=self.config.get('image_size', 512),
            interpolation=self.config.get('interpolation', 'bicubic'),
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            use_regularization=self.config.get('use_regularization', False)
        )

        # Initialize Model
        self.model = ConceptAblationModel(
            config_path=self.config.get('config_path'),
            ckpt_path=self.config.get('ckpt_path'),
            device=str(self.device)
        )

        # Initialize Trainer
        self.trainer = ConceptAblationTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            data_handler=self.data_handler
        )

    def run(self):
        """
        Execute the training process.
        """
        # Initialize WandB if needed
        project_name = self.config.get('project_name', 'concept_ablation_project')
        run_name = self.config.get('run_name', 'concept_ablation_run')
        wandb.init(project=project_name, name=run_name, config=self.config)
        self.logger.info("Initialized WandB for logging.")

        # Start training
        trained_model = self.trainer.train()

        # Save the trained model
        output_name = self.config.get('output_name', 'concept_ablation_model.pth')
        self.model.save_model(output_name)
        self.logger.info(f"Trained model saved at {output_name}")
        wandb.save(output_name)

        # Finish WandB run
        wandb.finish()
