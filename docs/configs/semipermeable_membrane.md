### Train Config
```python
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

```
