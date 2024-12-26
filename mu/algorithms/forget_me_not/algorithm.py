# forget_me_not/algorithm.py

import logging
from typing import Dict

import torch
import wandb
from algorithms.forget_me_not.data_handler import ForgetMeNotDataHandler
from algorithms.forget_me_not.model import ForgetMeNotModel
from algorithms.forget_me_not.trainer import ForgetMeNotTrainer


class ForgetMeNotAlgorithm:
    """
    Algorithm class orchestrating the Forget Me Not unlearning process.
    Handles both textual inversion (TI) and attention-based unlearning steps.
    """

    def __init__(self, config: Dict):
        """
        Initialize the ForgetMeNotAlgorithm.

        Args:
            config (Dict): Configuration dictionary containing all parameters required for training.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_components()

    def _setup_components(self):
        """
        Setup data handler, model, and trainer.
        """
        self.logger.info("Setting up components for Forget Me Not Algorithm...")
        self.data_handler = ForgetMeNotDataHandler(self.config)
        # Setup data if needed
        self.data_handler.setup()

        # Initialize model
        self.model = ForgetMeNotModel(self.config)

        # Initialize trainer
        self.trainer = ForgetMeNotTrainer(config=self.config, data_handler=self.data_handler, model=self.model, device=self.device)

    def run_ti_training(self):
        """
        Run the Textual Inversion (TI) training step.
        Corresponds to the logic in `train_ti.py`.
        """
        self.logger.info("Starting TI Training...")
        if self.config.get('use_wandb', False):
            wandb.init(project=self.config.get('wandb_project', 'forget_me_not'),
                       name=self.config.get('wandb_name', 'ti_run'),
                       config=self.config)
        self.trainer.train_ti()
        if self.config.get('use_wandb', False):
            wandb.finish()

    def run_attn_training(self):
        """
        Run the attention-based training step.
        Corresponds to the logic in `train_attn.py`.
        """
        self.logger.info("Starting Attention Training...")
        if self.config.get('use_wandb', False):
            wandb.init(project=self.config.get('wandb_project', 'forget_me_not'),
                       name=self.config.get('wandb_name', 'attn_run'),
                       config=self.config)
        self.trainer.train_attn()
        if self.config.get('use_wandb', False):
            wandb.finish()
