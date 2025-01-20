import os
from mu.core.base_config import BaseConfig


class SemipermeableMembraneConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Pretrained model configuration
        self.pretrained_model = {
            "name_or_path": "CompVis/stable-diffusion-v1-4",  # Model path or name
            "ckpt_path": "CompVis/stable-diffusion-v1-4",  # Checkpoint path
            "v2": False,  # Version 2 of the model
            "v_pred": False,  # Version prediction
            "clip_skip": 1,  # Skip layers in CLIP model
        }

        # Network configuration
        self.network = {
            "rank": 1,  # Network rank
            "alpha": 1.0,  # Alpha parameter for the network
        }

        # Training configuration
        self.train = {
            "precision": "float32",  # Training precision (e.g., "float32" or "float16")
            "noise_scheduler": "ddim",  # Noise scheduler method
            "iterations": 3000,  # Number of training iterations
            "batch_size": 1,  # Batch size
            "lr": 0.0001,  # Learning rate for the model
            "unet_lr": 0.0001,  # Learning rate for UNet
            "text_encoder_lr": 5e-05,  # Learning rate for text encoder
            "optimizer_type": "AdamW8bit",  # Optimizer type (e.g., "AdamW", "AdamW8bit")
            "lr_scheduler": "cosine_with_restarts",  # Learning rate scheduler type
            "lr_warmup_steps": 500,  # Steps for learning rate warm-up
            "lr_scheduler_num_cycles": 3,  # Number of cycles for the learning rate scheduler
            "max_denoising_steps": 30,  # Max denoising steps (for DDIM)
        }

        # Save configuration
        self.save = {
            "per_steps": 500,  # Save model every N steps
            "precision": "float32",  # Precision for saving model
        }

        # Other settings
        self.other = {
            "use_xformers": True  # Whether to use memory-efficient attention with xformers
        }

        # Weights and Biases (wandb) configuration
        self.wandb_project = "semipermeable_membrane_project"  # wandb project name
        self.wandb_run = "spm_run"  # wandb run name

        # Dataset configuration
        self.use_sample = True  # Use sample dataset for training
        self.dataset_type = (
            "unlearncanvas"  # Dataset type (e.g., "unlearncanvas", "i2p")
        )
        self.template = "style"  # Template type (e.g., "style", "object")
        self.template_name = "Abstractionism"  # Template name

        # Prompt configuration
        self.prompt = {
            "target": self.template_name,  # Prompt target (can use the template name)
            "positive": self.template_name,  # Positive prompt (can use the template name)
            "unconditional": "",  # Unconditional prompt
            "neutral": "",  # Neutral prompt
            "action": "erase_with_la",  # Action to perform (e.g., "erase_with_la")
            "guidance_scale": "1.0",  # Guidance scale for generation
            "resolution": 512,  # Image resolution
            "batch_size": 1,  # Batch size for prompt generation
            "dynamic_resolution": True,  # Flag for dynamic resolution
            "la_strength": 1000,  # Strength of the latent attention (la)
            "sampling_batch_size": 4,  # Batch size for sampling
        }

        # Device configuration
        self.devices = "0"  # CUDA devices to train on (comma-separated)

        # Output configuration
        self.output_dir = "outputs/semipermeable_membrane/finetuned_models"  # Directory to save models

        # Verbose logging
        self.verbose = True  # Whether to log verbose information during training

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # Check if necessary directories exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Validate pretrained model paths
        if not os.path.exists(self.pretrained_model["ckpt_path"]):
            raise FileNotFoundError(
                f"Checkpoint path {self.pretrained_model['ckpt_path']} does not exist."
            )

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
        if self.train["iterations"] <= 0:
            raise ValueError("iterations should be a positive integer.")
        if self.train["batch_size"] <= 0:
            raise ValueError("batch_size should be a positive integer.")
        if self.train["lr"] <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.train["unet_lr"] <= 0:
            raise ValueError("UNet learning rate (unet_lr) should be positive.")
        if self.train["text_encoder_lr"] <= 0:
            raise ValueError(
                "Text encoder learning rate (text_encoder_lr) should be positive."
            )
        if self.train["lr_warmup_steps"] < 0:
            raise ValueError("lr_warmup_steps should be non-negative.")
        if self.train["max_denoising_steps"] <= 0:
            raise ValueError("max_denoising_steps should be positive.")

        # Validate output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Validate WandB project and run names
        if not isinstance(self.wandb_project, str) or not isinstance(
            self.wandb_run, str
        ):
            raise ValueError("wandb_project and wandb_run should be strings.")

        # Validate prompt configuration
        if not isinstance(self.prompt["action"], str):
            raise ValueError("Action should be a string.")

        if not isinstance(self.prompt["guidance_scale"], str):
            raise ValueError("guidance_scale should be a string.")


semipermiable_membrane_train_config_quick_canvas = SemipermeableMembraneConfig()
semipermiable_membrane_train_config_quick_canvas.dataset_type = "unlearncanvas"
semipermiable_membrane_train_config_quick_canvas.raw_dataset_dir = (
    "datasets/quick-canvas-dataset/sample"
)


semipermiable_membrane_train_config_i2p = SemipermeableMembraneConfig()
semipermiable_membrane_train_config_i2p.dataset_type = "i2p"
semipermiable_membrane_train_config_i2p.raw_dataset_dir = "datasets/i2p-dataset/sample"
