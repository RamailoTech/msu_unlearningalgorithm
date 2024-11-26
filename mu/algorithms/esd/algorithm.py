from core.base_algorithm import BaseAlgorithm
from algorithms.esd.model import ESDModel
from algorithms.esd.trainer import ESDTrainer
from algorithms.esd.sampler import ESDSampler
import torch
import wandb
from typing import Dict

class ESDAlgorithm(BaseAlgorithm):
    """
    ESD Algorithm for machine unlearning.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.trainer = None
        self.sampler = None
        self.device = torch.device(self.config.get('devices', ['cuda:0'])[0])
        self.device_orig = torch.device(self.config.get('devices', ['cuda:0'])[1])
        self._setup_components()

    def _setup_components(self):
        print(self.config)
        self.model = ESDModel(self.config.get('config_path'), self.config.get('ckpt_path'), self.device)
        self.trainer = ESDTrainer(self.model, self.config, self.device, self.device_orig)
        self.sampler = ESDSampler(self.model, self.config, self.device)

    def run(self):
        # Initialize wandb
        wandb.init(project='quick-canvas-machine-unlearning', name=self.config['theme'], config=self.config)

        # Train the model
        self.trainer.train()

        # Save the model
        output_name = self.config['output_name']
        self.model.save_model(output_name)
        print(f"Model saved to {output_name}")
