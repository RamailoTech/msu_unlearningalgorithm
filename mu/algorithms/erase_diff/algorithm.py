# mu/algorithms/erase_diff/algorithm.py

import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.erase_diff.model import EraseDiffModel
from mu.algorithms.erase_diff.trainer import EraseDiffTrainer
from mu.algorithms.erase_diff.data_handler import EraseDiffDataHandler
from mu.algorithms.erase_diff.configs import EraseDiffConfig


class EraseDiffAlgorithm(BaseAlgorithm):
    """
    EraseDiffAlgorithm orchestrates the training process for the EraseDiff method.
    """

    def __init__(self, config: EraseDiffConfig, **kwargs):
        """
        Initialize the EraseDiffAlgorithm.

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
        self.device = torch.device(self.config.get("devices", ["cuda:0"])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _parse_config(self):
        self.config["lr"] = float(self.config["lr"])
        return super()._parse_config()

    def _setup_components(self):
        """
        Setup model, data handler, trainer, and sampler components.
        """
        self.logger.info("Setting up components...")
        # Initialize Data Handler

        self.data_handler = EraseDiffDataHandler(
            raw_dataset_dir=self.config.get("raw_dataset_dir"),
            processed_dataset_dir=self.config.get("processed_dataset_dir"),
            dataset_type=self.config.get("dataset_type", "unlearncanvas"),
            template=self.config.get("template"),
            template_name=self.config.get("template_name"),
            batch_size=self.config.get("batch_size", 4),
            image_size=self.config.get("image_size", 512),
            interpolation=self.config.get("interpolation", "bicubic"),
            use_sample=self.config.get("use_sample", False),
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
        )

        # Initialize Model
        self.model = EraseDiffModel(
            model_config_path=self.config.get("model_config_path"),
            ckpt_path=self.config.get("ckpt_path"),
            device=str(self.device),
        )

        # Initialize Trainer
        self.trainer = EraseDiffTrainer(
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
                "name": self.config.get("wandb_run", "EraseDiff"),
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
                    "output_name",
                    f"erase_diff_{self.config.get('template_name')}_model.pth",
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
