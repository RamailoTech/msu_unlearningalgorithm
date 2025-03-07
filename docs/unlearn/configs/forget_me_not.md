### Train Ti Config
```python
class ForgetMeNotTiConfig(BaseConfig):
    """
    Configuration class for the Forget-Me-Not textual inversion training.
    Mirrors the fields from the second YAML snippet.
    """

    def __init__(self, **kwargs):
        # Model checkpoint path
        self.ckpt_path = "models/diffuser/style50"

        # Dataset directories
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/forget_me_not/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"
        self.use_sample = True  # Use the sample dataset for training

        # Training configuration
        self.initializer_tokens = self.template_name
        self.steps = 10
        self.lr = 1e-4
        self.weight_decay_ti = 0.1
        self.seed = 42
        self.placeholder_tokens = "<s1>|<s2>|<s3>|<s4>"
        self.placeholder_token_at_data = "<s>|<s1><s2><s3><s4>"
        self.gradient_checkpointing = False
        self.scale_lr = False
        self.gradient_accumulation_steps = 1
        self.train_batch_size = 1
        self.lr_warmup_steps = 100

        # Output configuration
        self.output_dir = "outputs/forget_me_not/ti_models"

        # Device configuration
        self.devices = "0"  # CUDA devices to train on (comma-separated)

        # Additional configurations
        self.tokenizer_name = "default_tokenizer"
        self.instance_prompt = "default_prompt"
        self.concept_keyword = "default_keyword"
        self.lr_scheduler = "linear"
        self.prior_generation_precision = "fp32"
        self.local_rank = 0
        self.class_prompt = "default_class_prompt"
        self.num_class_images = 100
        self.dataloader_num_workers = 4
        self.center_crop = True
        self.prior_loss_weight = 0.1
```

### Train Attn config
```python

class ForgetMeNotAttnConfig(BaseConfig):
    """
    This class encapsulates the training configuration for the 'Forget-Me-Not' TI approach.
    It mirrors the fields specified in the YAML-like config snippet.
    """

    def __init__(self, **kwargs):
        # Model and checkpoint paths
        self.ckpt_path = "models/diffuser/style50"

        # Dataset directories and setup
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/forget_me_not/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"
        self.use_sample = True  # Use the sample dataset for training

        # Textual Inversion config
        self.use_ti = True
        self.ti_weights_path = "outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors"
        self.initializer_tokens = self.template_name
        self.placeholder_tokens = "<s1>|<s2>|<s3>|<s4>"

        # Training configuration
        self.mixed_precision = None  # or "fp16", if desired
        self.gradient_accumulation_steps = 1
        self.train_text_encoder = False
        self.enable_xformers_memory_efficient_attention = False
        self.gradient_checkpointing = False
        self.allow_tf32 = False
        self.scale_lr = False
        self.train_batch_size = 1
        self.use_8bit_adam = False
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 0.01
        self.adam_epsilon = 1.0e-08
        self.size = 512
        self.with_prior_preservation = False
        self.num_train_epochs = 1
        self.lr_warmup_steps = 0
        self.lr_num_cycles = 1
        self.lr_power = 1.0
        self.max_steps = 2  # originally "max-steps" in config
        self.no_real_image = False
        self.max_grad_norm = 1.0
        self.checkpointing_steps = 500
        self.set_grads_to_none = False
        self.lr = 5e-5

        # Output configurations
        self.output_dir = "outputs/forget_me_not/finetuned_models/Abstractionism"

        # Device configuration
        self.devices = "0"  # CUDA devices to train on (comma-separated)
        self.only_xa = True  # originally "only-xa" in config

        # Additional 'Forget-Me-Not' parameters
        self.perform_inversion = True
        self.continue_inversion = True
        self.continue_inversion_lr = 0.0001
        self.learning_rate_ti = 0.001
        self.learning_rate_unet = 0.0003
        self.learning_rate_text = 0.0003
        self.lr_scheduler = "constant"
        self.lr_scheduler_lora = "linear"
        self.lr_warmup_steps_lora = 0
        self.prior_loss_weight = 1.0
        self.weight_decay_lora = 0.001
        self.use_face_segmentation_condition = False
        self.max_train_steps_ti = 500
        self.max_train_steps_tuning = 1000
        self.save_steps = 100
        self.class_data_dir = None
        self.stochastic_attribute = None
        self.class_prompt = None
        self.num_class_images = 100
        self.resolution = 512
        self.color_jitter = False
        self.sample_batch_size = 1
        self.lora_rank = 4
        self.clip_ti_decay = True
```


### Description of Arguments in train_ti_config.yaml

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`). Use `generic` as type if you want to use your own dataset. Valid choices are `unlearncanvas`, `i2p` and `generic`.
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Training Configuration**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **steps**: Number of training steps.
- **lr**: Learning rate for the training optimizer.
- **weight_decay_ti**: Weight decay for Text Inversion training.
- **seed**: Random seed for reproducibility.
- **placeholder_tokens**: Tokens used as placeholders during training.
- **placeholder_token_at_data**: Placeholders used in the dataset for Text Inversion training.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_batch_size**: Batch size for training.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).



### Description of Arguments in train_attn_config.yaml

### Key Parameters

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Text Inversion**

- **use_ti**: Boolean indicating whether to use Text Inversion weights.
- **ti_weights_path**: File path to the Text Inversion model weights.

**Tokens**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **placeholder_tokens**: Tokens used as placeholders during training.

**Training Configuration**

- **mixed_precision**: Precision type to use during training (e.g., `fp16` or `fp32`).
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_text_encoder**: Boolean to enable or disable training of the text encoder.
- **enable_xformers_memory_efficient_attention**: Boolean to enable memory-efficient attention mechanisms.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **allow_tf32**: Boolean to allow TensorFloat-32 computation for faster training.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **train_batch_size**: Batch size for training.
- **use_8bit_adam**: Boolean to enable or disable 8-bit Adam optimizer.
- **adam_beta1**: Beta1 parameter for the Adam optimizer.
- **adam_beta2**: Beta2 parameter for the Adam optimizer.
- **adam_weight_decay**: Weight decay for the Adam optimizer.
- **adam_epsilon**: Epsilon value for the Adam optimizer.
- **size**: Image resolution size for training.
- **with_prior_preservation**: Boolean indicating whether to use prior preservation during training.
- **num_train_epochs**: Number of training epochs.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.
- **lr_num_cycles**: Number of cycles for learning rate scheduling.
- **lr_power**: Exponent to control the shape of the learning rate curve.
- **max-steps**: Maximum number of training steps.
- **no_real_image**: Boolean to skip using real images in training.
- **max_grad_norm**: Maximum norm for gradient clipping.
- **checkpointing_steps**: Number of steps between model checkpoints.
- **set_grads_to_none**: Boolean to set gradients to None instead of zeroing them out.
- **lr**: Learning rate for the training optimizer.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).

**Miscellaneous**

- **only-xa**: Boolean to enable additional configurations specific to the XA pipeline.



