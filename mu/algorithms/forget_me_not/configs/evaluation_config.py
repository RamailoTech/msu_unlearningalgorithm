# mu/algorithms/forget_me_not/configs/evaluation_config.py

import os

from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class ForgetMeNotEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.ckpt_path = "outputs/forget_me_not/finetuned_models"  # path to finetuned model checkpoint directory
        self.cfg_text_list = [9.0]  # list of classifier-free guidance scales
        self.seed = 188  # random seed
        self.ddim_steps = 100  # number of DDIM steps
        self.image_height = 512  # height of the image
        self.image_width = 512  # width of the image
        self.ddim_eta = 0.0  # DDIM eta parameter
        self.devices = "0"  # GPU device ID
        self.sampler_output_dir = "outputs/eval_results/mu_results/forget_me_not/"  # directory to save sampler outputs
        self.dataset_type = "unlearncanvas"
        self.use_sample = True


        # Override defaults with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint directory {self.ckpt_path} does not exist.")
        if not os.path.exists(self.sampler_output_dir):
            os.makedirs(self.sampler_output_dir)
        if self.dataset_type not in ["unlearncanvas", "i2p", "generic"]:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        if any(cfg <= 0 for cfg in self.cfg_text_list):
            raise ValueError("Classifier-free guidance scale (cfg_text) values should be positive.")
        if self.ddim_steps <= 0:
            raise ValueError("DDIM steps should be a positive integer.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Image height and width should be positive.")


# Example usage
forget_me_not_evaluation_config = ForgetMeNotEvaluationConfig()