# mu/algorithms/semipermeable_membrane/configs/train_config.py

import os

from mu.core.base_config import BaseConfig


class PretrainedModelConfig(BaseConfig):
    def __init__(
        self,
        name_or_path="CompVis/stable-diffusion-v1-4",  # Model path or name
        ckpt_path="CompVis/stable-diffusion-v1-4",  # Checkpoint path
        v2=False,  # Version 2 of the model
        v_pred=False,  # Version prediction
        clip_skip=1,  # Skip layers in CLIP model
    ):
        self.name_or_path = name_or_path
        self.ckpt_path = ckpt_path
        self.v2 = v2
        self.v_pred = v_pred
        self.clip_skip = clip_skip


class NetworkConfig(BaseConfig):
    def __init__(
        self,
        rank=1,  # Network rank
        alpha=1.0,  # Alpha parameter for the network
    ):
        self.rank = rank
        self.alpha = alpha


class TrainConfig(BaseConfig):
    def __init__(
        self,
        precision="float32",  # Training precision (e.g., "float32" or "float16")
        noise_scheduler="ddim",  # Noise scheduler method
        iterations=3000,  # Number of training iterations
        batch_size=1,  # Batch size
        lr=0.0001,  # Learning rate for the model
        unet_lr=0.0001,  # Learning rate for UNet
        text_encoder_lr=5e-05,  # Learning rate for text encoder
        optimizer_type="AdamW8bit",  # Optimizer type (e.g., "AdamW", "AdamW8bit")
        lr_scheduler="cosine_with_restarts",  # Learning rate scheduler type
        lr_warmup_steps=500,  # Steps for learning rate warm-up
        lr_scheduler_num_cycles=3,  # Number of cycles for the learning rate scheduler
        max_denoising_steps=30,  # Max denoising steps (for DDIM)
    ):
        self.precision = precision
        self.noise_scheduler = noise_scheduler
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr = lr
        self.unet_lr = unet_lr
        self.text_encoder_lr = text_encoder_lr
        self.optimizer_type = optimizer_type
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_scheduler_num_cycles = lr_scheduler_num_cycles
        self.max_denoising_steps = max_denoising_steps


class SaveConfig(BaseConfig):
    def __init__(
        self,
        per_steps=500,  # Save model every N steps
        precision="float32",  # Precision for saving model
    ):
        self.per_steps = per_steps
        self.precision = precision


class OtherConfig(BaseConfig):
    def __init__(
        self,
        use_xformers=True,  # Whether to use memory-efficient attention with xformers
    ):
        self.use_xformers = use_xformers


class PromptConfig(BaseConfig):
    def __init__(
        self,
        target="Abstractionism",  # Prompt target
        positive="Abstractionism",  # Positive prompt
        unconditional="",  # Unconditional prompt
        neutral="",  # Neutral prompt
        action="erase_with_la",  # Action to perform
        guidance_scale="1.0",  # Guidance scale for generation
        resolution=512,  # Image resolution
        batch_size=1,  # Batch size for prompt generation
        dynamic_resolution=True,  # Flag for dynamic resolution
        la_strength=1000,  # Strength of the latent attention
        sampling_batch_size=4,  # Batch size for sampling
    ):
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral
        self.action = action
        self.guidance_scale = guidance_scale
        self.resolution = resolution
        self.batch_size = batch_size
        self.dynamic_resolution = dynamic_resolution
        self.la_strength = la_strength
        self.sampling_batch_size = sampling_batch_size


class SemipermeableMembraneConfig(BaseConfig):
    """
    SemipermeableMembraneConfig stores all the configuration parameters for the
    semipermeable membrane training, including model, network, training, saving,
    and other environment details.
    """

    def __init__(self, **kwargs):
        # Pretrained model configuration
        self.pretrained_model = PretrainedModelConfig()

        # Network configuration
        self.network = NetworkConfig()

        # Training configuration
        self.train = TrainConfig()

        # Save configuration
        self.save = SaveConfig()

        # Other settings
        self.other = OtherConfig()

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
        self.prompt = PromptConfig(
            target=self.template_name,
            positive=self.template_name,
        )

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

        # Validate dataset type
        if self.dataset_type not in ["unlearncanvas", "i2p"]:
            raise ValueError(
                f"Invalid dataset type {self.dataset_type}. Choose from ['unlearncanvas', 'i2p']"
            )

        # Validate training settings
        if self.train.iterations <= 0:
            raise ValueError("iterations should be a positive integer.")
        if self.train.batch_size <= 0:
            raise ValueError("batch_size should be a positive integer.")
        if self.train.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.train.unet_lr <= 0:
            raise ValueError("UNet learning rate (unet_lr) should be positive.")
        if self.train.text_encoder_lr <= 0:
            raise ValueError(
                "Text encoder learning rate (text_encoder_lr) should be positive."
            )
        if self.train.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps should be non-negative.")
        if self.train.max_denoising_steps <= 0:
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
        if not isinstance(self.prompt.action, str):
            raise ValueError("Action should be a string.")
        if not isinstance(self.prompt.guidance_scale, str):
            raise ValueError("guidance_scale should be a string.")


semipermiable_membrane_train_mu = SemipermeableMembraneConfig()
semipermiable_membrane_train_mu.dataset_type = "unlearncanvas"

semipermiable_membrane_train_i2p = SemipermeableMembraneConfig()
semipermiable_membrane_train_i2p.dataset_type = "i2p"
