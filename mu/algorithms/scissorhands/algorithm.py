import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.scissorhands.model import ScissorHandsModel
from mu.algorithms.scissorhands.trainer import ScissorHandsTrainer
from mu.algorithms.scissorhands.data_handler import ScissorHandsDataHandler
from mu.algorithms.scissorhands.configs import ScissorHandsConfig


class ScissorHandsAlgorithm(BaseAlgorithm):
    """
    ScissorhandsAlgorithm orchestrates the training process for the Scissorhands method.
    """

    def __init__(self, config: ScissorHandsConfig, **kwargs):
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        self._parse_config()
        config.validate_config()
        self.device = torch.device(self.config.get("devices", ["cuda:0"])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _parse_config(self):
        return super()._parse_config()

    def _setup_components(self):
        self.logger.info("Setting up components...")
        # Initialize components

        self.data_handler = ScissorHandsDataHandler(
            raw_dataset_dir=self.config.get("raw_dataset_dir"),
            processed_dataset_dir=self.config.get("processed_dataset_dir"),
            dataset_type=self.config.get("dataset_type", "unlearncanvas"),
            template=self.config.get("template"),
            template_name=self.config.get("template_name"),
            batch_size=self.config.get("batch_size", 4),
            image_size=self.config.get("image_size", 512),
            interpolation=self.config.get("interpolation", "bicubic"),
            use_sample=self.config.get("use_sample", False),
        )
        # Initialize Model
        self.model = ScissorHandsModel(
            model_config_path=self.config.get("model_config_path"),
            ckpt_path=self.config.get("ckpt_path"),
            device=str(self.device),
        )

        # Initialize Trainer
        self.trainer = ScissorHandsTrainer(
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
                "name": self.config.get("wandb_run", "scissorhands"),
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
                    f"scissorhands_{self.config.get('template_name')}_model.pth",
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
