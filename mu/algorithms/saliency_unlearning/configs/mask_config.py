# mu/algorithms/saliency_unlearning/configs/mask_config.py

import os

from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class SaliencyUnlearningMaskConfig(BaseConfig):
    def __init__(self, **kwargs):
        
        self.c_guidance = 7.5
        self.batch_size = 1
        self.num_timesteps = 1
        self.image_size =  512

        self.model_config_path = current_dir / "model_config.yaml"
        self.ckpt_path =  "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion

        # Dataset directories
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/saliency_unlearning/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"
        self.threshold = 0.5

        # Directory Configuration
        self.output_dir = "outputs/saliency_unlearning/masks"  # Output directory to save results

        # Training Configuration
        self.lr = 0.00001
        self.devices = "0"  # CUDA devices to train on (comma-separated)
        self.use_sample = True

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # Check if directories exist
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            os.makedirs(self.processed_dataset_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")



saliency_unlearning_generate_mask_mu = SaliencyUnlearningMaskConfig()
saliency_unlearning_generate_mask_mu.dataset_type = "unlearncanvas"
saliency_unlearning_generate_mask_mu.raw_dataset_dir = "data/quick-canvas-dataset/sample"

saliency_unlearning_generate_mask_i2p = SaliencyUnlearningMaskConfig()
saliency_unlearning_generate_mask_i2p.dataset_type = "i2p"
saliency_unlearning_generate_mask_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
