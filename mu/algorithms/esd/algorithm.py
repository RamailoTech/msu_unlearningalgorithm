import torch
import wandb
from typing import Dict
from pathlib import Path
import logging

from mu.core import BaseAlgorithm
from mu.algorithms.esd.model import ESDModel
from mu.algorithms.esd.trainer import ESDTrainer
from mu.algorithms.esd.sampler import ESDSampler
from mu.algorithms.esd.configs import ESDConfig


class ESDAlgorithm(BaseAlgorithm):
    """
    ESD Algorithm for machine unlearning.
    """

    def __init__(self, config: ESDConfig, **kwargs):
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)

        self._parse_config()
        config.validate_config()
        self.model = None
        self.trainer = None
        self.sampler = None
        self.device = torch.device(self.config.get("devices", ["cuda:0"])[0])
        self.device_orig = torch.device(self.config.get("devices", ["cuda:0"])[1])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _parse_config(self):
        template_name = self.config.get("template_name", "")
        prompt = f"An image of {template_name}."
        self.config["prompt"] = prompt
        return super()._parse_config()

    def _setup_components(self):
        """
        Setup model, trainer, and sampler components.
        """
        self.logger.info("Setting up components...")

        self.model = ESDModel(
            self.config.get("model_config_path"),
            self.config.get("ckpt_path"),
            self.device,
            self.device_orig,
        )
        self.sampler = ESDSampler(self.model, self.config, self.device)
        self.trainer = ESDTrainer(
            self.model, self.sampler, self.config, self.device, self.device_orig
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
                "name": self.config.get("wandb_run", "ESD"),
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
                    "output_name", f"esd_{self.config.get('template_name')}_model.pth"
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
