import os
from mu.core.base_config import BaseConfig


class ForgetMeNotConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Common Configurations
        self.ckpt_path = "models/diffuser/style50"
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/forget_me_not/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"  # Anchor for template
        self.use_sample = True  # Use the sample dataset for training

        # Training configurations (common)
        self.mixed_precision = None
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
        self.max_steps = 2
        self.no_real_image = False
        self.max_grad_norm = 1.0
        self.checkpointing_steps = 500
        self.set_grads_to_none = False
        self.lr = 5e-5

        # Output configuration (common)
        self.output_dir = "outputs/forget_me_not/finetuned_models/Abstractionism"
        self.devices = "0"

        # Perform inversion
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

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory {self.processed_dataset_dir} does not exist."
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


class ForgetMeNotAttnConfig(ForgetMeNotConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Attn-specific settings (for forget-me-not attention model)
        self.only_xa = True
        self.lr_warmup_steps_lora = 100  # Different warmup for lora
        self.gradient_checkpointing = False

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class ForgetMeNotTiConfig(ForgetMeNotConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TI-specific settings (for forget-me-not inversion model)
        self.lr = 1e-4
        self.weight_decay_ti = 0.1
        self.seed = 42
        self.placeholder_tokens = "<s1>|<s2>|<s3>|<s4>"
        self.placeholder_token_at_data = "<s>|<s1><s2><s3><s4>"

        # Training configuration for TI-specific
        self.steps = 10
        self.gradient_accumulation_steps = 1
        self.train_batch_size = 1
        self.lr_warmup_steps = 100

        # Output configurations for TI
        self.output_dir = "outputs/forget_me_not/ti_models"
        self.devices = "0"

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the TI-specific config parameters.
        """
        super().validate_config()

        if self.steps <= 0:
            raise ValueError("Steps should be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) for TI should be positive.")
        if self.weight_decay_ti < 0:
            raise ValueError("Weight decay for TI should be non-negative.")
