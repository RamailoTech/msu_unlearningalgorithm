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

### Description of arguments being used in train_config class

These are the configuration used for training a Stable Diffusion model using the ESD (Erase Stable Diffusion) method. It defines various parameters related to training, model setup, dataset handling, and output configuration. Below is a detailed description of each section and parameter:

**Training Parameters**

These parameters control the fine-tuning process, including the method of training, guidance scales, learning rate, and iteration settings.

* train_method: Specifies the method of training to decide which parts of the model to update.

    * Type: str
    * Choices: noxattn, selfattn, xattn, full, notime, xlayer, selflayer
    * Example: xattn

* start_guidance: Guidance scale for generating initial images during training. Affects the diversity of the training set.

    * Type: float
    * Example: 0.1

* negative_guidance: Guidance scale for erasing the target concept during training.

    * Type: float
    * Example: 0.0

* iterations: Number of training iterations (similar to epochs).

    * Type: int
    * Example: 1

* lr: Learning rate used by the optimizer for fine-tuning.

    * Type: float
    * Example: 5e-5

* image_size: Size of images used during training and sampling (in pixels).

    * Type: int
    * Example: 512

* ddim_steps: Number of diffusion steps used in the DDIM sampling process.

    * Type: int
    * Example: 50


**Model Configuration**

These parameters specify the Stable Diffusion model checkpoint and configuration file.

* model_config_path: Path to the YAML file defining the model architecture and parameters.

    * Type: str
    * Example: mu/algorithms/esd/configs/model_config.yaml

* ckpt_path: Path to the finetuned Stable Diffusion model checkpoint.

    * Type: str
    * Example: '../models/compvis/style50/compvis.ckpt'


**Dataset Configuration**

These parameters define the dataset type and template for training, specifying whether to focus on objects, styles, or inappropriate content.

* dataset_type: Type of dataset used for training. Use `generic` as type if you want to use your own dataset.

    * Type: str
    * Choices: unlearncanvas, i2p, generic
    * Example: unlearncanvas

* template: Type of concept or style to erase during training.

    * Type: str
    * Choices: object, style, i2p
    * Example: style

* template_name: Specific name of the object or style to erase (e.g., "Abstractionism").

    * Type: str
    * Example Choices: Abstractionism, self-harm
    * Example: Abstractionism


**Output Configuration**

These parameters control where the outputs of the training process, such as fine-tuned models, are stored.

* output_dir: Directory where the fine-tuned model and training results will be saved.

    * Type: str
    * Example: outputs/esd/finetuned_models

* separator: Separator character used to handle multiple prompts during training. If set to null, no special handling occurs.

    * Type: str or null
    * Example: null

**Device Configuration**

These parameters define the compute resources for training.

* devices: Specifies the CUDA devices used for training. Provide a comma-separated list of device IDs.

    * Type: str
    * Example: 0,1

* use_sample: Boolean flag indicating whether to use a sample dataset for testing or debugging.

    * Type: bool
    * Example: True



