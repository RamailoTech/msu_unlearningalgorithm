import os
from mu.core.base_config import BaseConfig


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

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # Check if directories exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Validate model checkpoint path
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint path {self.ckpt_path} does not exist.")

        # Validate dataset type
        if self.dataset_type not in ["unlearncanvas", "i2p"]:
            raise ValueError(
                f"Invalid dataset type {self.dataset_type}. Choose from ['unlearncanvas', 'i2p']"
            )

        # Validate devices
        devices = self.devices.split(",")
        for device in devices:
            if not device.isdigit():
                raise ValueError(
                    f"Invalid device {device}. Devices should be integers representing CUDA device IDs."
                )

        # Validate training settings
        if self.epochs <= 0:
            raise ValueError("epochs should be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("alpha should be between 0 and 1.")

        # Validate the technique used
        if self.technique not in ["replace", "tensor"]:
            raise ValueError("Invalid technique. Choose from ['replace', 'tensor'].")

        # Validate preserve scale and erase scale
        if self.preserve_scale < 0 or self.preserve_scale > 1:
            raise ValueError("preserve_scale should be between 0 and 1.")
        if self.erase_scale <= 0:
            raise ValueError("erase_scale should be positive.")

        # Validate lamb
        if self.lamb < 0:
            raise ValueError("lamb should be a positive value.")

        # Validate preserver_concepts
        if not isinstance(self.preserver_concepts, str):
            raise ValueError("preserver_concepts should be a string.")

        # Validate base model
        if self.base not in ["stable-diffusion-v1-4"]:
            raise ValueError(
                f"Invalid base model {self.base}. Expected 'stable-diffusion-v1-4'."
            )


unified_concept_editing_train_config_quick_canvas = UnifiedConceptEditingConfig()
unified_concept_editing_train_config_quick_canvas.dataset_type = "unlearncanvas"

unified_concept_editing_train_config_i2p = UnifiedConceptEditingConfig()
unified_concept_editing_train_config_i2p.dataset_type = "i2p"
