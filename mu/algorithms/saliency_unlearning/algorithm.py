import os
import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.saliency_unlearning.model import SaliencyUnlearnModel
from mu.algorithms.saliency_unlearning.trainer import SaliencyUnlearnTrainer
from mu.algorithms.saliency_unlearning.data_handler import SaliencyUnlearnDataHandler
from mu.algorithms.saliency_unlearning.masking import (
    accumulate_gradients_for_mask,
    save_mask,
)
from mu.algorithms.saliency_unlearning.configs import SaliencyUnlearningConfig


class SaliencyUnlearnAlgorithm(BaseAlgorithm):
    """
    SaliencyUnlearnAlgorithm orchestrates the training process for the SaliencyUnlearn method.
    """

    def __init__(self, config: SaliencyUnlearningConfig, **kwargs):
        """
        Initialize the SaliencyUnlearnAlgorithm.

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
        return super()._parse_config()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up SaliencyUnlearn components...")

        # Initialize Data Handler
        self.data_handler = SaliencyUnlearnDataHandler(
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
            mask_path=self.config.get("mask_path"),
            use_mask=True,
        )

        mask_path = self.config.get("mask_path")

        mask = None
        if mask_path is not None:
            mask = torch.load(mask_path)

        # Initialize Model
        self.model = SaliencyUnlearnModel(
            model_config_path=self.config.get("model_config_path"),
            ckpt_path=self.config.get("ckpt_path"),
            mask=mask,
            device=str(self.device),
        )

        # Initialize Trainer
        self.trainer = SaliencyUnlearnTrainer(
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
                "name": self.config.get("wandb_run", "saliency_unlearn"),
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
                    f"saliency_unlearning_{self.config.get('template_name')}_model.pth",
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


class MaskingAlgorithm(BaseAlgorithm):
    """
    MaskingAlgorithm sets up the model and data for generating a saliency mask.
    It uses the same structure as the training algorithm but runs a single pass
    to accumulate gradients and then creates a mask.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.data_handler = None
        self.device = torch.device(self.config.get("devices", ["cuda:0"])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        self.logger.info("Setting up components for MaskingAlgorithm...")
        # Initialize Data Handler
        self.data_handler = SaliencyUnlearnDataHandler(
            raw_dataset_dir=self.config.get("raw_dataset_dir"),
            processed_dataset_dir=self.config.get("processed_dataset_dir"),
            dataset_type=self.config.get("dataset_type", "unlearncanvas"),
            template=self.config.get("template"),
            template_name=self.config.get("template_name"),
            mask_path=None,
            batch_size=self.config.get("batch_size", 4),
            image_size=self.config.get("image_size", 512),
            interpolation=self.config.get("interpolation", "bicubic"),
            use_sample=self.config.get("use_sample", False),
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
        )

        # Initialize Model
        self.model = SaliencyUnlearnModel(
            model_config_path=self.config.get("model_config_path"),
            ckpt_path=self.config.get("ckpt_path"),
            mask={},
            device=str(self.device),
        )

    def run(self):
        """
        Run the mask generation process:
        - Get the forget DataLoader
        - Accumulate gradients to create a mask
        - Save the mask to a .pt file
        """
        data_loaders = self.data_handler.get_data_loaders()
        forget_dl = data_loaders.get("forget")

        prompt = self.config.get(
            "prompt", f"An image in {self.config.get('theme', '')} Style."
        )
        c_guidance = self.config.get("c_guidance", 7.5)
        lr = self.config.get("lr", 1e-5)
        num_timesteps = self.config.get("num_timesteps", 1000)
        threshold = self.config.get("threshold", 0.5)
        batch_size = self.config.get("batch_size", 4)

        # Accumulate gradients and create mask
        mask = accumulate_gradients_for_mask(
            model=self.model,
            forget_loader=forget_dl,
            prompt=prompt,
            c_guidance=c_guidance,
            device=self.device,
            lr=lr,
            num_timesteps=num_timesteps,
            threshold=threshold,
            batch_size=batch_size,
        )

        # Save mask
        output_dir = self.config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        mask_path = os.path.join(output_dir, f"{threshold}.pt")
        save_mask(mask, mask_path)

        self.logger.info(f"Mask saved at {mask_path}")
