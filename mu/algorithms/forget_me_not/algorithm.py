# mu/algorithms/mu/forget_me_not/algorithm.py

import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.forget_me_not.data_handler import ForgetMeNotDataHandler
from mu.algorithms.forget_me_not.trainer import ForgetMeNotTrainer
from mu.algorithms.forget_me_not.model import ForgetMeNotModel
from mu.algorithms.forget_me_not.configs import ForgetMeNotTiConfig


class ForgetMeNotAlgorithm(BaseAlgorithm):
    """
    Algorithm class orchestrating the Forget Me Not unlearning process.
    Handles both textual inversion (TI) and attention-based unlearning steps.
    """

    def __init__(self, config: ForgetMeNotTiConfig, train_type="train_ti", **kwargs):
        """
        Initialize the ForgetMeNotAlgorithm.

        Args:
            config (Dict): Configuration dictionary containing all parameters required for training.
        """
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        print(f"Config = {self.config}")
        config.validate_config()
        self.type = train_type
        self._parse_config()
        self.model = None
        self.trainer = None
        self.data_handler = None
        self.logger = logging.getLogger(__name__)
        self.device = self._get_device()
        self._setup_components()

    def _parse_config(self):
        self.config["type"] = self.type
        return super()._parse_config()

    def _get_device(self):
        """
        Determine the device to use for training.
        """
        devices = self.config.get("devices", ["cuda:0"])
        if torch.cuda.is_available():
            return torch.device(devices[0])
        return torch.device("cpu")

    def _setup_components(self):
        """
        Setup data handler, model, and trainer.
        """
        self.logger.info("Setting up components for Forget Me Not Algorithm...")

        # Initialize model
        self.model = ForgetMeNotModel(self.config)

        self.data_handler = ForgetMeNotDataHandler(
            config=self.config, tokenizer=self.model.tokenizer
        )

        # Initialize trainer
        self.trainer = ForgetMeNotTrainer(
            config=self.config,
            data_handler=self.data_handler,
            model=self.model,
            device=self.device,
        )

    def run_ti_training(self):
        """
        Run the Textual Inversion (TI) training step.
        Corresponds to the logic in `train_ti.py`.
        """
        self.logger.info("Starting TI Training...")
        self.trainer.train_ti()

    def run_attn_training(self):
        """
        Run the attention-based training step.
        Corresponds to the logic in `train_attn.py`.
        """
        self.logger.info("Starting Attention Training...")
        self.trainer.train_attn()

    def run(self, train_type: str, *args, **kwargs):
        """
        Execute the Forget Me Not unlearning process.
        """
        try:
            # Initialize WandB with configurable project/run names
            wandb_config = {
                "project": self.config.get(
                    "wandb_project", "quick-canvas-machine-unlearning"
                ),
                "name": self.config.get("wandb_run", "Forget Me Not"),
                "config": self.config,
            }
            wandb.init(**wandb_config)
            self.logger.info("Initialized WandB for logging.")

            # Create output directory if it doesn't exist
            output_dir = Path(self.config.get("output_dir", "./outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                if train_type == "train_ti":
                    self.run_ti_training()
                elif train_type == "train_attn":
                    self.run_attn_training()
                else:
                    raise ValueError(
                        f"Invalid training type: {train_type}. Please choose either 'train_ti' or 'train_attn'."
                    )

                output_name = output_dir / self.config.get(
                    "output_name",
                    f"forget_me_not_{self.config.get('template_name')}_model.pth",
                )

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
