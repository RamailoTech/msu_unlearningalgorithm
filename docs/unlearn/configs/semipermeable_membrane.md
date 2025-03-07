### Train Config
```python
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


```

### Description of Arguments in train_config.yaml

**pretrained_model**

* ckpt_path: File path to the pretrained model's checkpoint file.

* v2: Boolean indicating whether the pretrained model is version 2 or not.

* v_pred: Boolean to enable/disable "v-prediction" mode for diffusion models.

* clip_skip: Number of CLIP layers to skip during inference.

**network**

* rank: Rank of the low-rank adaptation network.

* alpha: Scaling factor for the network during training.


**train**

* precision: Numerical precision to use during training (e.g., float32 or float16).

* noise_scheduler: Type of noise scheduler to use in the training loop (e.g., ddim).

* iterations: Number of training iterations.

* batch_size: Batch size for training.

* lr: Learning rate for the training optimizer.

* unet_lr: Learning rate for the U-Net model.

* text_encoder_lr: Learning rate for the text encoder.

* optimizer_type: Optimizer to use for training (e.g., AdamW8bit).

* lr_scheduler: Learning rate scheduler to apply during training.

* lr_warmup_steps: Number of steps for linear warmup of the learning rate.

* lr_scheduler_num_cycles: Number of cycles for a cosine-with-restarts scheduler.

* max_denoising_steps: Maximum denoising steps to use during training.

**save**

* per_steps: Frequency of saving the model (in steps).

* precision: Numerical precision for saved model weights


**other**

* use_xformers: Boolean to enable xformers memory-efficient attention.

* wandb_project and wandb_run

* Configuration for tracking the training progress using Weights & Biases.

* wandb_project: Project name in W&B.

* wandb_run: Specific run name in the W&B dashboard.

**use_sample**

* Boolean to indicate whether to use the sample dataset for training.

**template**

* Specifies the template type, choices are:
    * object: Focus on specific objects.
    * style: Focus on artistic styles.
    * i2p: Intermediate style processing.

**template_name**

* Name of the template, choices are:
    * self-harm
    * Abstractionism

**prompt**

* target: Target template or concept to guide training (references template_name).

* positive: Positive prompt based on the template.

* unconditional: Unconditional prompt text.

* neutral: Neutral prompt text.

* action: Specifies the action applied to the prompt (e.g., erase_with_la).

* guidance_scale: Guidance scale for classifier-free guidance.

* resolution: Image resolution for training.

* batch_size: Batch size for generating prompts.

* dynamic_resolution: Boolean to allow dynamic resolution.

* la_strength: Strength of local adaptation.

* sampling_batch_size: Batch size for sampling images.

**devices**

* CUDA devices to use for training (specified as a comma-separated list, e.g., "0,1").

**output_dir**

* Directory to save the fine-tuned model and other outputs.

**verbose**

* Boolean flag for verbose logging during training.



