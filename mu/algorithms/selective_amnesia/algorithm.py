# mu/algorithms/selective_amnesia/algorithm.py

import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.selective_amnesia.model import SelectiveAmnesiaModel
from mu.algorithms.selective_amnesia.trainer import SelectiveAmnesiaTrainer

class SelectiveAmnesiaAlgorithm(BaseAlgorithm):
    """
    Orchestrates the Selective Amnesia training process.
    Sets up model, data handler, and trainer, then runs training.
    """

    def __init__(self, config: Dict, config_path: str):
        self.config = config
        self.config_path = config_path
        self.model = None
        self.trainer = None
        self.device = self.config.get('devices')
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        self.logger.info("Setting up components...")

        # Model: Load SD model and FIM
        # Initialize Model
        self.model = SelectiveAmnesiaModel(
            model_config_path=self.config.get('model_config_path'),
            ckpt_path=self.config.get('ckpt_path'),
            device=str(self.device),
            opt_config=self.config
        )

        # Initialize Trainer
        self.trainer = SelectiveAmnesiaTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            config_path = self.config_path
        )

    def run(self):
        try:
            # Initialize WandB with configurable project/run names
            wandb_config = {
                "project": self.config.get("wandb_project", "quick-canvas-machine-unlearning"),
                "name": self.config.get("wandb_run", "Selective Amnesia"),
                "config": self.config
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
