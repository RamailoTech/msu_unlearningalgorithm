import logging
from typing import Dict

import torch
import wandb
from algorithms.scissorhands.data_handler import ScissorHandsDataHandler
from algorithms.scissorhands.model import ScissorHandsModel
from algorithms.scissorhands.trainer import ScissorHandsTrainer
from core.base_algorithm import BaseAlgorithm


class ScissorHandsAlgorithm(BaseAlgorithm):
    """
    ScissorhandsAlgorithm orchestrates the training process for the Scissorhands method.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(self.config.get("devices", ["cuda:0"])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        self.logger.info("Setting up components...")
        # Initialize components
        self.data_handler = ScissorHandsDataHandler(
            original_data_dir=self.config.get("original_data_dir"),
            new_data_dir=self.config.get("new_data_dir"),
            selected_theme=self.config.get("theme"),
            selected_class=self.config.get("class"),
            batch_size=self.config.get("batch_size", 4),
            image_size=self.config.get("image_size", 512),
            interpolation=self.config.get("interpolation", "bicubic"),
            use_sample=self.config.get("use_sample", False),
        )
        # Initialize Model
        self.model = ScissorHandsModel(
            config_path=self.config.get("config_path"),
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
        # Initialize WandB
        wandb.init(
            project="scissorhands_unlearning",
            name=self.config.get("theme", "Scissorhands"),
            config=self.config,
        )
        self.logger.info("Initialized WandB for logging.")

        # Start training
        trained_model = self.trainer.train()

        # Save the trained model
        output_path = self.config.get("output_name", "scissorhands_model.pth")
        self.model.save_model(output_path)
        wandb.save(output_path)

        # Finish WandB run
        wandb.finish()
