# mu/algorithms/unified_concept_editing/algorithm.py

import torch
import wandb
import logging
from typing import Dict
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.unified_concept_editing.model import UnifiedConceptEditingModel
from mu.algorithms.unified_concept_editing.trainer import UnifiedConceptEditingTrainer
from mu.algorithms.unified_concept_editing.data_handler import (
    UnifiedConceptEditingDataHandler,
)
from mu.algorithms.unified_concept_editing.configs import UnifiedConceptEditingConfig


class UnifiedConceptEditingAlgorithm(BaseAlgorithm):
    """
    UnifiedConceptEditingAlgorithm orchestrates the training process for the Unified Concept Editing method.
    """

    def __init__(self, config: UnifiedConceptEditingConfig, **kwargs):
        """
        Initialize the UnifiedConceptEditingAlgorithm.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)

        self._parse_config()
        config.validate_config()
        self.model = None
        self.trainer = None
        self.data_handler = None
        self.device = torch.device(self.config.devices[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _parse_config(self):
        return super()._parse_config()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up components...")

        # Initialize Data Handler
        self.data_handler = UnifiedConceptEditingDataHandler(
            dataset_type=self.config.get("dataset_type", "unlearncanvas"),
            template=self.config.get("template"),
            template_name=self.config.get("template_name"),
            use_sample=self.config.get("use_sample", False),
        )

        # Initialize Model
        self.model = UnifiedConceptEditingModel(
            ckpt_path=self.config.get("ckpt_path"), device=str(self.device)
        )

        # Initialize Trainer
        self.trainer = UnifiedConceptEditingTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            data_handler=self.data_handler,
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
                "name": self.config.get("wandb_run", "UnifiedConceptEditing"),
                "config": self.config,
            }
            wandb.init(**wandb_config)
            self.logger.info("Initialized WandB for logging.")

            # Create output directory if it doesn't exist
            output_dir = Path(self.config.get("output_dir", "./outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Start training
                model = self.trainer.train()

                # Save final model
                output_name = output_dir / self.config.get(
                    "output_name", f"uce_{self.config.get('template_name')}_model.pth"
                )
                self.model.save_model(model, output_name)
                self.logger.info(f"Trained model saved at {output_name}")

                # Save to WandB
                wandb.save(str(output_name))

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
