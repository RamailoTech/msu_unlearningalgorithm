# mu/algorithms/concept_ablation/algorithm.py

import torch
import wandb
from typing import Dict
import logging
from pathlib import Path

from mu.core import BaseAlgorithm
from mu.algorithms.concept_ablation.data_handler import ConceptAblationDataHandler
from mu.algorithms.concept_ablation.model import ConceptAblationModel
from mu.algorithms.concept_ablation.trainer import ConceptAblationTrainer

class ConceptAblationAlgorithm(BaseAlgorithm):
    """
    ConceptAblationAlgorithm orchestrates the training process for the Concept Ablation method.
    It sets up the model, data handler, and trainer, and then runs the training loop.
    """

    def __init__(self, config: Dict):
        """
        Initialize the ConceptAblationAlgorithm.

        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.model = None
        self.trainer = None
        self.data_handler = None
        self.device = torch.device(self.config.get('devices', ['cuda:0'])[0])
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self):
        """
        Setup model, data handler, and trainer components.
        """
        self.logger.info("Setting up components...")

        # Initialize Data Handler
        self.data_handler = ConceptAblationDataHandler(
            concept_type=self.config.get('concept_type'),
            prompts_path=self.config.get('prompts_path'),
            output_dir=self.config.get('output_dir'),
            base_config=self.config.get('config_path'),
            resume_ckpt=self.config.get('ckpt_path'),
            delta_ckpt=self.config.get('delta_ckpt', None),
            caption_target=self.config.get('caption_target', None),
            train_size=self.config.get('train_size', 1000),
            n_samples=self.config.get('n_samples', 10),
            image_size=self.config.get('image_size', 512),
            interpolation=self.config.get('interpolation', 'bicubic'),
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            use_regularization=self.config.get('use_regularization', False)
        )

        # Initialize Model
        self.model = ConceptAblationModel(
            model_config_path=self.config.get('model_config_path'),
            ckpt_path=self.config.get('ckpt_path'),
            device=str(self.device)
        )

        # Initialize Trainer
        self.trainer = ConceptAblationTrainer(
            model=self.model,
            config=self.config,
            device=str(self.device),
            data_handler=self.data_handler
        )

    def run(self):
        """
        Execute the training process.
        """
        try:
            # Initialize WandB with configurable project/run names
            wandb_config = {
                "project": self.config.get("wandb_project", "quick-canvas-machine-unlearning"),
                "name": self.config.get("wandb_run", "Concept Ablation"),
                "config": self.config
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
                output_name = output_dir / self.config.get("output_name", f"concept_ablation_{self.config.get('template_name')}_model.pth")
                self.model.save_model(model,output_name)
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