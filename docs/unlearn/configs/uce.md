### Train Config
```python
class UnifiedConceptEditingConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Training configuration
        self.train_method = "full"  # Options: full, partial
        self.alpha = 0.1  # Guidance factor for training
        self.epochs = 1  # Number of epochs
        self.lr = 5e-5  # Learning rate

        # Model configuration
        self.ckpt_path = "models/diffuser/style50"  # Path to model checkpoint

        # Output configuration
        self.output_dir = (
            "outputs/uce/finetuned_models"  # Directory to save finetuned models
        )
        self.dataset_type = "unlearncanvas"  # Type of dataset to be used
        self.template = "style"  # Template for training
        self.template_name = "Abstractionism"  # Name of the template

        # Device configuration
        self.devices = "0"  # CUDA devices to train on (comma-separated)

        # Additional flags
        self.use_sample = True  # Whether to use the sample dataset

        # Editing-specific configuration
        self.guided_concepts = (
            "A Elephant image"  # Comma-separated string of guided concepts
        )
        self.technique = (
            "replace"  # Technique for editing (Options: "replace", "tensor")
        )

        # Parameters for the editing technique
        self.preserve_scale = 0.1  # Scale for preserving the concept (float)
        self.preserve_number = (
            None  # Number of concepts to preserve (int, None for all)
        )
        self.erase_scale = 1  # Scale for erasing
        self.lamb = 0.1  # Regularization weight for loss
        self.add_prompts = False  # Whether to add additional prompts

        # Preserver concepts (comma-separated if multiple)
        self.preserver_concepts = (
            "A Lion image"  # Comma-separated string of preserver concepts
        )

        # Base model used for editing
        self.base = "stable-diffusion-v1-4"  # Base version of Stable Diffusion
```



### Evaluation config

```python
# mu/algorithms/unified_concept_editing/configs/evaluation_config.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class UceEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.ckpt_path = "outputs/uce/finetuned_models/uce_Abstractionism_model.pth"  # path to finetuned model checkpoint
        self.pipeline_path = "UnlearnCanvas/machine_unlearning/models/diffuser/style50"  # path to pretrained pipeline
        self.classifier_ckpt_path = "models/classifier_ckpt_path/style50_cls.pth"  # path to classifier checkpoint
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
        self.classification_model = "vit_large_patch16_224"  # classification model for evaluation
        self.eval_output_dir = "outputs/eval_results/mu_results/uce"  # directory to save evaluation results
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
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline directory {self.pipeline_path} does not exist.")
        if not os.path.exists(self.classifier_ckpt_path):
            raise FileNotFoundError(f"Classifier checkpoint file {self.classifier_ckpt_path} does not exist.")
        if not os.path.exists(self.reference_dir):
            raise FileNotFoundError(f"Reference directory {self.reference_dir} does not exist.")
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir)
        if not os.path.exists(self.sampler_output_dir):
            os.makedirs(self.sampler_output_dir)

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
```