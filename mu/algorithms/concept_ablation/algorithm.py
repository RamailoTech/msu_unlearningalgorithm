# mu/algorithms/concept_ablation/algorithm.py

import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.core.base_config import BaseConfig
from mu.algorithms.concept_ablation.model import ConceptAblationModel
from mu.algorithms.concept_ablation.trainer import ConceptAblationTrainer
from mu.algorithms.concept_ablation.configs import ConceptAblationConfig


class ConceptAblationAlgorithm(BaseAlgorithm):
    """
    ConceptAblationAlgorithm orchestrates the training process for the Concept Ablation method.
    It sets up the model, data handler, and trainer, and then runs the training loop.
    """

    def __init__(self, config: ConceptAblationConfig, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(config, key):
                setattr(config, key, value)
                continue
            config_attr = getattr(config, key)
            if isinstance(config_attr, BaseConfig) and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    setattr(config_attr, sub_key, sub_val)
            elif isinstance(config_attr, dict) and isinstance(value, dict):
                config_attr.update(value)
            else: 
                setattr(config, key, value)
        self.config = config.to_dict()

        config.validate_config()
        self.config = config.to_dict()
        self.config_path = self.config.get("config_path")
        self.model = None
        self.trainer = None
        self.device = self.config.get("devices")[0]
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up components...")

        # Initialize Model
        self.model = ConceptAblationModel(
            train_config=self.config,
            model_config_path=self.config.get("model_config_path"),
            ckpt_path=self.config.get("ckpt_path"),
            device=str(self.device),
        )

        # Initialize Trainer
        self.trainer = ConceptAblationTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            config_path=self.config_path,
        )

    def run(self):
        """
        Execute the training process.
        """
        try:
            # Initialize WandB with configurable project/run names
            wandb_config = {
                "project": self.config.get(
                    "wandb_project", "quick-canvas-machine-unlearning"
                ),
                "name": self.config.get("wandb_run", "Concept Ablation"),
                "config": self.config,
            }
            wandb.init(**wandb_config)
            self.logger.info("Initialized WandB for logging.")

            # Create output directory if it doesn't exist
            output_dir = Path(self.config.get("output_dir", "./outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Start training
                self.trainer.train()

            except Exception as e:
                self.logger.error(f"Error during training: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Failed to initialize training: {str(e)}")
            raise

        finally:
            # Ensure WandB always finishes
            if wandb.run is not None:
                wandb.finish()
            self.logger.info("Training complete. WandB logging finished.")
