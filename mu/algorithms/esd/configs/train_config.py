import os
from mu.core.base_config import BaseConfig


class ESDConfig(BaseConfig):

    def __init__(self, **kwargs):
        # Training parameters
        self.train_method = "xattn"  # Choices: ["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"]
        self.start_guidance = (
            0.1  # Optional: guidance of start image (previously alpha)
        )
        self.negative_guidance = 0.0  # Optional: guidance of negative training
        self.iterations = 1  # Optional: iterations used to train (previously epochs)
        self.lr = 1e-5  # Optional: learning rate
        self.image_size = 512  # Optional: image size used to train
        self.ddim_steps = 50  # Optional: DDIM steps of inference

        # Model configuration
        self.model_config_path = "mu/algorithms/esd/configs/model_config.yaml"
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion

        # Dataset directories
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/esd/data"
        self.dataset_type = "unlearncanvas"  # Choices: ['unlearncanvas', 'i2p']
        self.template = "style"  # Choices: ['object', 'style', 'i2p']
        self.template_name = (
            "Abstractionism"  # Choices: ['self-harm', 'Abstractionism']
        )

        # Output configurations
        self.output_dir = "outputs/esd/finetuned_models"
        self.separator = None

        # Device configuration
        self.devices = "0,0"
        self.use_sample = True

        # For backward compatibility
        self.interpolation = "bicubic"  # Interpolation method
        self.ddim_eta = 0.0  # Eta for DDIM
        self.num_workers = 4  # Number of workers for data loading
        self.pin_memory = True  # Pin memory for faster transfer to GPU

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        TRAIN_METHODS = [
            "noxattn",
            "selfattn",
            "xattn",
            "full",
            "notime",
            "xlayer",
            "selflayer",
        ]
        if self.iterations <= 0:
            raise ValueError("iterations should be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.image_size <= 0:
            raise ValueError("Image size should be positive.")
        if self.train_method not in TRAIN_METHODS:
            raise ValueError(f"Invalid train method. Choose from {TRAIN_METHODS}")
        if self.dataset_type not in ["unlearncanvas", "i2p"]:
            raise ValueError(
                "Invalid dataset type. Choose from ['unlearncanvas', 'i2p']"
            )

        # Check if directories exist
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


esd_train_config_quick_canvas = ESDConfig()
esd_train_config_quick_canvas.dataset_type = "unlearncanvas"
esd_train_config_quick_canvas.raw_dataset_dir = "data/quick-canvas-dataset/sample"

esd_train_config_i2p = ESDConfig()
esd_train_config_i2p.dataset_type = "i2p"
esd_train_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
