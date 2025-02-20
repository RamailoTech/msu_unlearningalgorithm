# mu/algorithms/adv_unlearn/algorithm.py

import wandb
import logging

from pathlib import Path

from mu.core.base_config import BaseConfig
from mu.core import BaseAlgorithm
from mu_defense.algorithms.adv_unlearn import AdvUnlearnModel
from mu_defense.algorithms.adv_unlearn.trainer import AdvUnlearnTrainer
from mu_defense.algorithms.adv_unlearn.configs import AdvUnlearnConfig


class AdvUnlearnAlgorithm(BaseAlgorithm):
    """
    AdvUnlearnAlgorithm orchestrates the adversarial unlearning training process.
    It sets up the model and trainer components and then runs the training loop.
    """

    def __init__(self, config: AdvUnlearnConfig, **kwargs):
        # Update configuration with additional kwargs.
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

        # Validate and update config.
        config.validate_config()
        self.config = config.to_dict()
        self.model = None
        self.trainer = None
        self.devices = self.config.get("devices")
        self.devices = [f'cuda:{int(d.strip())}' for d in self.devices.split(',')]
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """
        Setup model and trainer components.
        """
        self.logger.info("Setting up components for adversarial unlearning training...")

        # Initialize Model
        self.model = AdvUnlearnModel(config=self.config)

        # Initialize Trainer
        self.trainer = AdvUnlearnTrainer(
            model=self.model,
            config=self.config,
            devices=self.devices,
            
        )
        self.trainer.trainer.adv_attack.model_orig = self.model.model_orig
        self.trainer.trainer.adv_attack.sampler_orig = self.model.sampler_orig
        self.trainer.trainer.adv_attack.model = self.model.model
        self.trainer.trainer.adv_attack.sampler = self.model.sampler

    def run(self):
        """
        Execute the training process.
        """
        try:
            # Initialize WandB with configurable project/run names.
            wandb_config = {
                "project": self.config.get("wandb_project", "adv-unlearn-project"),
                "name": self.config.get("wandb_run", "Adv Unlearn Training"),
                "config": self.config,
            }
            wandb.init(**wandb_config)
            self.logger.info("Initialized WandB for logging.")

            # Create output directory if it doesn't exist.
            output_dir = Path(self.config.get("output_dir", "./outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Start training.
                self.trainer.run()
            except Exception as e:
                self.logger.error(f"Error during training: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Failed to initialize training: {str(e)}")
            raise

        finally:
            # Ensure WandB always finishes.
            if wandb.run is not None:
                wandb.finish()
            self.logger.info("Training complete. WandB logging finished.")
