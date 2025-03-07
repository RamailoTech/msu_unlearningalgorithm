# mu/algorithms/unified_concept_editing/configs/evaluation_config.py

import os

from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class UceEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.ckpt_path = "outputs/uce/finetuned_models/uce_Abstractionism_model"  # path to finetuned model checkpoint
        # self.pipeline_path = "UnlearnCanvas/machine_unlearning/models/diffuser/style50"  # path to pretrained pipeline
        self.cfg_text = 9.0  # classifier-free guidance scale
        self.seed = 188  # random seed
        self.task = "class"  # task type
        self.devices = "0"  # GPU device ID
        self.ddim_steps = 100  # number of DDIM steps
        self.image_height = 512  # height of the image
        self.image_width = 512  # width of the image
        self.ddim_eta = 0.0  # DDIM eta parameter
        self.sampler_output_dir = "outputs/eval_results/mu_results/uce"  # directory to save sampler outputs
        self.seed_list = ["188"]  # list of seeds for evaluation
        self.batch_size = 1  # batch size for evaluation
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
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if not os.path.exists(self.sampler_output_dir):
            os.makedirs(self.sampler_output_dir)
        if self.dataset_type not in ["unlearncanvas", "i2p", "generic"]:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        if self.cfg_text <= 0:
            raise ValueError("Classifier-free guidance scale (cfg_text) should be positive.")
        if self.ddim_steps <= 0:
            raise ValueError("DDIM steps should be a positive integer.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Image height and width should be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size should be a positive integer.")
        if self.task not in ["class", "other_task"]:  # Add other valid tasks if needed
            raise ValueError("Invalid task type.")


# Example usage
uce_evaluation_config = UceEvaluationConfig()