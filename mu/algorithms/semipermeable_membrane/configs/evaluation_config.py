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
        self.model_ckpt_path = "CompVis/stable-diffusion-v1-4"  # path to the base model checkpoint
        self.theme = "Bricks"  # theme for evaluation
        self.seed = 188  # random seed
        self.devices = "0"  # GPU device ID
        self.task = "class"  # task type
        self.sampler_output_dir = "outputs/eval_results/mu_results/semipermeable_membrane/"  # directory to save sampler outputs
        self.seed_list = [188]  # list of seeds for evaluation
        self.classification_model = "vit_large_patch16_224"  # classification model for evaluation
        self.eval_output_dir = "outputs/eval_results/mu_results/semipermeable_membrane/"  # directory to save evaluation results
        self.reference_dir = "data/quick-canvas-dataset/sample/"  # path to the original dataset
        self.forget_theme = "Bricks"  # theme to forget
        self.multiprocessing = False  # whether to use multiprocessing

        # Override defaults with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # if not os.path.exists(self.model_config_path):
        #     raise FileNotFoundError(f"Model config directory {self.model_config_path} does not exist.")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if not os.path.exists(self.reference_dir):
            raise FileNotFoundError(f"Reference directory {self.reference_dir} does not exist.")
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir)
        if not os.path.exists(self.sampler_output_dir):
            os.makedirs(self.sampler_output_dir)

        if any(multiplier <= 0 for multiplier in self.spm_multiplier):
            raise ValueError("SPM multiplier values should be positive.")
        if self.task not in ["class", "other_task"]:  # Add other valid tasks if needed
            raise ValueError("Invalid task type.")


# Example usage
semipermeable_membrane_eval_config = SemipermeableMembraneEvaluationConfig()