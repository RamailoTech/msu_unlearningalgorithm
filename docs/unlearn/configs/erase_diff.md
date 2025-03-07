### Sample Train Config
```python
class EraseDiffConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.train_method = "xattn"
        self.alpha = 0.1
        self.epochs = 1
        self.K_steps = 2
        self.lr = 5e-5
        self.model_config_path = current_dir / "model_config.yaml"
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/erase_diff/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"
        self.output_dir = "outputs/erase_diff/finetuned_models"
        self.separator = None
        self.image_size = 512
        self.interpolation = "bicubic"
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.devices = "0"
        self.use_sample = True
        self.num_workers = 4
        self.pin_memory = True


```


### Sample Model Config

```bash
model:

  base_learning_rate: 1.0e-04
  target: stable_diffusion.ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "edited"
    cond_stage_key: "edit"
    image_size: 64
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

### Description of Arguments in train_config.yaml

**Training Parameters**

* train_method: Specifies the method of training for concept erasure.

    * Choices: ["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"]
    * Example: "xattn"

* alpha: Guidance strength for the starting image during training.

    * Type: float
    * Example: 0.1

* epochs: Number of epochs to train the model.

    * Type: int
    * Example: 1

* K_steps: Number of K optimization steps during training.

    * Type: int
    * Example: 2

* lr: Learning rate used for the optimizer during training.

    * Type: float
    * Example: 5e-5

**Model Configuration**

* model_config_path: File path to the Stable Diffusion model configuration YAML file.

    * type: str
    * Example: "/path/to/model_config.yaml"

* ckpt_path: File path to the checkpoint of the Stable Diffusion model.

    * Type: str
    * Example: "/path/to/model_checkpoint.ckpt"


**Dataset Directories**

* raw_dataset_dir: Directory containing the raw dataset categorized by themes or classes.

    * Type: str
    * Example: "/path/to/raw_dataset"

* processed_dataset_dir: Directory to save the processed dataset.

    * Type: str
    * Example: "/path/to/processed_dataset"

* dataset_type: Specifies the dataset type for the training process. Use `generic` as type if you want to use your own dataset.

    * Choices: ["unlearncanvas", "i2p", "generic"]
    * Example: "unlearncanvas"

* template: Type of template to use during training.

    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* template_name: Name of the specific concept or style to be erased.

    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"


**Output Configurations**

* output_dir: Directory where the fine-tuned models and results will be saved.

    * Type: str
    * Example: "outputs/erase_diff/finetuned_models"

* separator: String separator used to train multiple words separately, if applicable.

    * Type: str or null
    * Example: null

**Sampling and Image Configurations**

* image_size: Size of the training images (height and width in pixels).

    * Type: int
    * Example: 512

* interpolation: Interpolation method used for image resizing.

    * Choices: ["bilinear", "bicubic", "lanczos"]
    * Example: "bicubic"

* ddim_steps: Number of DDIM inference steps during training.

    * Type: int
    * Example: 50

* ddim_eta: DDIM eta parameter for stochasticity during sampling.

    * Type: float
    * Example: 0.0

**Device Configuration**

* devices: Specifies the CUDA devices to be used for training (comma-separated).

    * Type: str
    * Example: "0"


**Additional Flags**

* use_sample: Flag to indicate whether a sample dataset should be used for training.

    * Type: bool
    * Example: True

* num_workers: Number of worker threads for data loading.

    * Type: int
    * Example: 4

* pin_memory: Flag to enable pinning memory during data loading for faster GPU transfers.

    * Type: bool
    * Example: true




