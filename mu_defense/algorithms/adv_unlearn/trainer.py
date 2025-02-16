# mu_defense/algorithms/adv_unlearn/trainer.py

import logging

from mu_defense.core import BaseTrainer
from mu_defense.algorithms.adv_unlearn import AdvUnlearnCompvisTrainer,AdvUnlearnDiffuserTrainer

class AdvUnlearnTrainer(BaseTrainer):
    """
    Trainer class orchestrates the adversarial unlearning training process.
    It instantiates the model and trainer components based on the provided configuration,
    and then runs the training loop.
    """
    def __init__(self, config: dict, model, devices):

        self.backend = config.get("backend")
        self.logger = logging.getLogger(__name__)
        
        # Setup components based on the backend.
        if self.backend == "compvis":
            self.logger.info("Using Compvis backend for adversarial unlearning.")
            self.trainer = AdvUnlearnCompvisTrainer(model, config, devices)
        if self.backend == "diffusers":
            self.logger.info("Using Diffusers backend for adversarial unlearning.")
            self.trainer = AdvUnlearnDiffuserTrainer(model, config, devices)
            

    def run(self):
        """
        Run the training loop.
        """
        self.logger.info("Starting training...")
        self.trainer.train()
        self.logger.info("Training complete.")


