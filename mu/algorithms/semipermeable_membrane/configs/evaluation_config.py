# mu/algorithms/semipermeable_membrane/configs/evaluation_config.py

import os

from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class SemipermeableMembraneEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.precision = "fp32"  # precision for computation
        self.spm_multiplier = [1.0]  # list of semipermeable membrane multipliers
        self.v2 = False  # whether to use version 2 of the model
        self.matching_metric = "clipcos_tokenuni"  # matching metric for evaluation
        self.model_config_path = "machine_unlearning/mu_semipermeable_membrane_spm/configs"  # path to model config
        self.base_model = "CompVis/stable-diffusion-v1-4"  # base model for the algorithm
        self.spm_path = ["outputs/semipermeable_membrane/finetuned_models/semipermeable_membrane_Abstractionism_last.safetensors"]  # path to semipermeable membrane model
        self.ckpt_path = "outputs/semipermeable_membrane/finetuned_models/semipermeable_membrane_Abstractionism_last.safetensors"  # path to finetuned model checkpoint
        # self.model_ckpt_path = "CompVis/stable-diffusion-v1-4"  # path to the base model checkpoint
        self.seed = 188  # random seed
        self.devices = "0"  # GPU device ID
        self.sampler_output_dir = "outputs/eval_results/mu_results/semipermeable_membrane/"  # directory to save sampler outputs
        self.forget_theme = "Bricks"  # theme to forget
        # self.multiprocessing = False  # whether to use multiprocessing
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

        if any(multiplier <= 0 for multiplier in self.spm_multiplier):
            raise ValueError("SPM multiplier values should be positive.")
        if self.task not in ["class", "other_task"]:  # Add other valid tasks if needed
            raise ValueError("Invalid task type.")


# Example usage
semipermeable_membrane_eval_config = SemipermeableMembraneEvaluationConfig()