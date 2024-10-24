from algorithms.core.base_algorithm import BaseAlgorithm
from algorithms.esd.esd_model import ESDModel
from algorithms.esd.esd_trainer import ESDTrainer
from algorithms.esd.esd_sampler import ESDSampler
import torch
import wandb
from typing import Dict

class ESDAlgorithm(BaseAlgorithm):
    def __init__(self, config: Dict):
        self.config = config
        self.devices = [torch.device(f'cuda:{int(d.strip())}') for d in config['devices'].split(',')]

        # Initialize models and samplers
        self.model_orig = ESDModel(config['config_path'], config['ckpt_path'], device=self.devices[1])
        self.sampler_orig = ESDSampler(self.model_orig)

        self.model = ESDModel(config['config_path'], config['ckpt_path'], device=self.devices[0])
        self.sampler = ESDSampler(self.model)

        # Initialize trainer
        self.trainer = ESDTrainer(
            model=self.model,
            optimizer=None,  # Will be set in trainer
            criterion=torch.nn.MSELoss(),
            config=config,
            sampler=self.sampler,
            model_orig=self.model_orig,
            sampler_orig=self.sampler_orig,
            wandb=wandb.init(project='unlearning_project', name=config['object_class'], config=config) if not config.get('dry_run', False) else None
        )

    def run(self):
        prompt = self.config['prompt']
        if self.config.get('seperator'):
            words = [word.strip() for word in prompt.split(self.config['seperator'])]
        else:
            words = [prompt]

        self.trainer.train(
            num_iterations=self.config['iterations'],
            words=words,
            train_method=self.config['train_method'],
            start_guidance=self.config['start_guidance'],
            negative_guidance=self.config['negative_guidance']
        )

        # Save the model
        output_name = f"{self.config['output_dir']}/{self.config['object_class']}.pth"
        self.model.save_model(output_name)
        print(f"Model saved to {output_name}")
