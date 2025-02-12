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


### Evaluation config

```python
# mu/algorithms/forget_me_not/configs/evaluation_config.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class ForgetMeNotEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.ckpt_path = "outputs/forget_me_not/finetuned_models"  # path to finetuned model checkpoint directory
        self.classifier_ckpt_path = "models/classifier_ckpt_path/style50_cls.pth"  # path to classifier checkpoint
        self.forget_theme = "Bricks"  # theme to forget (load finetuned model for this theme)
        self.cfg_text_list = [9.0]  # list of classifier-free guidance scales
        self.seed = 188  # random seed
        self.task = "class"  # task type
        self.ddim_steps = 100  # number of DDIM steps
        self.image_height = 512  # height of the image
        self.image_width = 512  # width of the image
        self.ddim_eta = 0.0  # DDIM eta parameter
        self.devices = "0"  # GPU device ID
        self.sampler_output_dir = "outputs/eval_results/mu_results/forget_me_not/"  # directory to save sampler outputs
        self.seed_list = ["188"]  # list of seeds for evaluation
        self.classification_model = "vit_large_patch16_224"  # classification model for evaluation
        self.eval_output_dir = "outputs/eval_results/mu_results/forget_me_not/"  # directory to save evaluation results
        self.reference_dir = "data/quick-canvas-dataset/sample/"  # path to the original dataset
        self.multiprocessing = False  # whether to use multiprocessing

        # Override defaults with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint directory {self.ckpt_path} does not exist.")
        if not os.path.exists(self.classifier_ckpt_path):
            raise FileNotFoundError(f"Classifier checkpoint file {self.classifier_ckpt_path} does not exist.")
        if not os.path.exists(self.reference_dir):
            raise FileNotFoundError(f"Reference directory {self.reference_dir} does not exist.")
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir)
        if not os.path.exists(self.sampler_output_dir):
            os.makedirs(self.sampler_output_dir)

        if any(cfg <= 0 for cfg in self.cfg_text_list):
            raise ValueError("Classifier-free guidance scale (cfg_text) values should be positive.")
        if self.ddim_steps <= 0:
            raise ValueError("DDIM steps should be a positive integer.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Image height and width should be positive.")
        if self.task not in ["class", "other_task"]:  # Add other valid tasks if needed
            raise ValueError("Invalid task type.")


# Example usage
forget_me_not_evaluation_config = ForgetMeNotEvaluationConfig()
```