# erasediff_algorithm.py

from algorithms.core.base_algorithm import BaseAlgorithm
from algorithms.erasediff.erasediff_model import EraseDiffModel
from algorithms.erasediff.erasediff_trainer import EraseDiffTrainer
from algorithms.erasediff.erasediff_data_handler import EraseDiffDataHandler
import torch
from typing import Dict
import wandb

class EraseDiffAlgorithm(BaseAlgorithm):
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.model = EraseDiffModel(config['config_path'], config['ckpt_path'], self.device)

        self.data_handler = EraseDiffDataHandler(
            forget_data_dir=config['forget_data_dir'],
            remain_data_dir=config['remain_data_dir'],
            batch_size=config.get('batch_size', 4),
            image_size=config.get('image_size', 512)
        )

        self.trainer = EraseDiffTrainer(
            model=self.model,
            optimizer=None,  # Will be set in trainer
            criterion=torch.nn.MSELoss(),
            config=config,
            data_handler=self.data_handler,
            wandb=wandb.init(project="quick-canvas-machine-unlearning", name=config['theme'], config=config) if not config.get('dry_run', False) else None
        )

    def run(self):
        self.trainer.train()
        # Save the trained model
        output_name = os.path.join(self.config['output_dir'], "sd.ckpt")
        self.model.save_model(output_name)
        print(f"Model saved to {output_name}")
