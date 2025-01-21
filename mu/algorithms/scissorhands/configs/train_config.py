import os
from mu.core.base_config import BaseConfig


class ScissorHandsConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Training parameters
        self.train_method = "xattn"  # choices: ["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"]
        self.alpha = 0.75  # Guidance of start image used to train
        self.epochs = 5  # Number of training epochs

        # Model configuration
        self.model_config_path = "mu/algorithms/scissorhands/configs/model_config.yaml"  # Config path for model
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion

        # Dataset directories
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/scissorhands/data"
        self.dataset_type = "unlearncanvas"  # Choices: ["unlearncanvas", "i2p"]
        self.template = "style"  # Template to use
        self.template_name = "Abstractionism"  # Template name

        # Output configuration
        self.output_dir = (
            "outputs/scissorhands/finetuned_models"  # Output directory to save results
        )

        # Sampling and image configurations
        self.sparsity = 0.90  # Threshold for mask sparsity
        self.project = False  # Whether to project
        self.memory_num = 1  # Number of memories to use
        self.prune_num = 10  # Number of pruned images

        # Device configuration
        self.devices = "0,1"  # CUDA devices to train on (comma-separated)

        # Additional configurations
        self.use_sample = True  # Use sample dataset for training

        # Guidance configurations
        self.start_guidence = 0.5  # Starting guidance factor
        self.negative_guidance = 0.3  # Negative guidance factor
        self.iterations = 1000  # Number of training iterations

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # Check if necessary directories exist
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory {self.processed_dataset_dir} does not exist."
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Check if the model config and checkpoint files exist
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model config file {self.model_config_path} does not exist."
            )
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")

        # Validate dataset type
        if self.dataset_type not in ["unlearncanvas", "i2p"]:
            raise ValueError(
                f"Invalid dataset type {self.dataset_type}. Choose from ['unlearncanvas', 'i2p']"
            )

        # Validate training method
        TRAIN_METHODS = [
            "noxattn",
            "selfattn",
            "xattn",
            "full",
            "notime",
            "xlayer",
            "selflayer",
        ]
        if self.train_method not in TRAIN_METHODS:
            raise ValueError(
                f"Invalid train method {self.train_method}. Choose from {TRAIN_METHODS}"
            )

        # Check if devices are valid
        devices = self.devices.split(",")
        for device in devices:
            if not device.isdigit():
                raise ValueError(
                    f"Invalid device {device}. Devices should be integers representing CUDA device IDs."
                )

        # Validate sparsity
        if not (0 <= self.sparsity <= 1):
            raise ValueError(
                f"Sparsity should be between 0 and 1. Given: {self.sparsity}"
            )

        # Validate iterations
        if self.iterations <= 0:
            raise ValueError(
                f"Iterations should be a positive integer. Given: {self.iterations}"
            )

        # Validate guidance values
        if not (0 <= self.start_guidence <= 1):
            raise ValueError(
                f"Start guidance should be between 0 and 1. Given: {self.start_guidence}"
            )
        if not (0 <= self.negative_guidance <= 1):
            raise ValueError(
                f"Negative guidance should be between 0 and 1. Given: {self.negative_guidance}"
            )


scissorhands_train_config_quick_canvas = ScissorHandsConfig()
scissorhands_train_config_quick_canvas.dataset_type = "unlearncanvas"
scissorhands_train_config_quick_canvas.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

scissorhands_train_config_i2p = ScissorHandsConfig()
scissorhands_train_config_i2p.dataset_type = "i2p"
scissorhands_train_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
