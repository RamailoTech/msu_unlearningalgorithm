# saliency_unlearning_algorithm.py

from typing import Dict

class SaliencyUnlearningAlgorithm(BaseAlgorithm):
    def __init__(self, config: Dict):
        self.config = config

        # Initialize DataHandler
        self.data_handler = SaliencyDataHandler(
            forget_data_dir=config['forget_data_dir'],
            remain_data_dir=config['remain_data_dir'],
            image_size=config.get('image_size', 512)
        )

        # Initialize Model
        self.model = SaliencyModel()
        self.model.load_model(
            config_path=config['config_path'],
            ckpt_path=config['ckpt_path'],
            device=config.get('device', 'cuda')
        )

        # Initialize Trainer
        self.trainer = SaliencyTrainer(
            model=self.model,
            config=config
        )

    def run(self):
        batch_size = self.config.get('batch_size', 4)
        forget_loader, remain_loader = self.data_handler.get_data_loaders(batch_size)

        prompt = self.config['prompt']
        c_guidance = self.config.get('c_guidance', 1.0)
        num_timesteps = self.config.get('num_timesteps', 1000)
        threshold = self.config.get('threshold', 0.5)

        # Compute the saliency mask
        self.trainer.compute_saliency_mask(
            forget_loader=forget_loader,
            prompt=prompt,
            c_guidance=c_guidance,
            num_timesteps=num_timesteps,
            threshold=threshold
        )

        epochs = self.config.get('epochs', 1)
        alpha = self.config.get('alpha', 0.1)
        self.trainer.train(
            forget_loader=forget_loader,
            remain_loader=remain_loader,
            epochs=epochs,
            alpha=alpha
        )

        output_path = self.config.get('output_path', 'output_model.ckpt')
        self.trainer.save_checkpoint(output_path)
