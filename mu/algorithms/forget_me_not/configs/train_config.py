import os
from mu.core.base_config import BaseConfig
from pathlib import Path

current_dir = Path(__file__).parent


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

        # Override default values if provided via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Optionally, implement any validations you need here.
        For instance, checking if directories exist, or if certain
        numerical hyperparameters are within a valid range.
        """
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(
                f"Directory '{self.raw_dataset_dir}' does not exist."
            )

        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory '{self.processed_dataset_dir}' does not exist."
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


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

        # Override defaults with any kwargs provided
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Add any custom validation logic for the configuration here.
        """
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(
                f"Directory '{self.raw_dataset_dir}' does not exist."
            )
        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory '{self.processed_dataset_dir}' does not exist."
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Example checks:
        if self.steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.train_batch_size <= 0:
            raise ValueError("Train batch size must be positive.")


ForgetMeNotAttnConfig

forget_me_not_train_ti_config_quick_canvas = ForgetMeNotAttnConfig()
forget_me_not_train_ti_config_quick_canvas.dataset_type = "unlearncanvas"
forget_me_not_train_ti_config_quick_canvas.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

forget_me_not_train_attn_config_quick_canvas = ForgetMeNotAttnConfig()
forget_me_not_train_attn_config_quick_canvas.dataset_type = "unlearncanvas"
forget_me_not_train_attn_config_quick_canvas.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

forget_me_not_train_ti_config_i2p = ForgetMeNotTiConfig()
forget_me_not_train_ti_config_i2p.dataset_type = "i2p"
forget_me_not_train_ti_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"

forget_me_not_train_attn_config_i2p = ForgetMeNotAttnConfig()
forget_me_not_train_attn_config_i2p.dataset_type = "i2p"
forget_me_not_train_attn_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
