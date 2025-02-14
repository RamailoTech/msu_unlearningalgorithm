### Sample Train Config

```python
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
        self.model_config_path = current_dir / "model_config.yaml"
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

```

### Sample Model Config
```yaml
model:
  base_learning_rate: 1.0e-04
  target: stable_diffusion.ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 32
    channels: 4
    cond_stage_trainable: false   # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    scheduler_config: # 10000 warmup steps
      target: stable_diffusion.ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 10000 ]
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]

    unet_config:
      target: stable_diffusion.ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: stable_diffusion.ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: stable_diffusion.ldm.modules.encoders.modules.FrozenCLIPEmbedder
```


### Evaluation config

```python
# mu/algorithms/esd/configs/evaluation_config.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class ESDEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.model_config_path = current_dir/"model_config.yaml"  # path to model config
        self.ckpt_path = "outputs/esd/finetuned_models/esd_Abstractionism_model.pth"  # path to finetuned model checkpoint
        self.classifier_ckpt_path = "models/classifier_ckpt_path/style50_cls.pth"  # path to classifier checkpoint
        self.forget_theme = "Bricks"  # theme to forget
        self.cfg_text = 9.0  # classifier-free guidance scale
        self.devices = "0"  # GPU device ID
        self.seed = 188  # random seed
        self.task = "class"  # task type
        self.ddim_steps = 100  # number of DDIM steps
        self.image_height = 512  # height of the image
        self.image_width = 512  # width of the image
        self.ddim_eta = 0.0  # DDIM eta parameter
        self.sampler_output_dir = "outputs/eval_results/mu_results/esd/"  # directory to save sampler outputs
        self.seed_list = ["188"]  # list of seeds for evaluation
        self.classification_model = "vit_large_patch16_224"  # classification model for evaluation
        self.eval_output_dir = "outputs/eval_results/mu_results/esd/"  # directory to save evaluation results
        self.reference_dir = "data/quick-canvas-dataset/sample/"  # path to the original dataset
        self.multiprocessing = False  # whether to use multiprocessing

        # Override defaults with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"Model config file {self.model_config_path} does not exist.")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
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
        if self.task not in ["class", "other_task"]:  # Add other valid tasks if needed
            raise ValueError("Invalid task type.")


# Example usage
esd_evaluation_config = ESDEvaluationConfig()
```