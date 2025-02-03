import os
from pathlib import Path
from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent

class AdvUnlearnConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Inference & Model Paths
        self.config_path = current_dir / "configs/stable-diffusion/v1-inference.yaml"
        self.ckpt_path = "models/sd-v1-4-full-ema.ckpt"
        self.diffusers_config_path = current_dir / "diffusers_unet_config.json"
        self.model_name_or_path = "CompVis/stable-diffusion-v1-4"
        self.cache_path = ".cache"
        
        # Devices & IO
        self.devices = "0,0"  # You can later parse this string into a list if needed.
        self.seperator = None
        self.output_dir = "outputs/adv_unlearn"
        
        # Image & Diffusion Sampling
        self.image_size = 512
        self.ddim_steps = 50
        self.start_guidance = 3.0
        self.negative_guidance = 1.0

        # Training Setup
        self.prompt = "nudity"
        self.dataset_retain = "coco"  # Choices: 'coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'
        self.retain_batch = 5
        self.retain_train = "iter"  # Options: 'iter' or 'reg'
        self.retain_step = 1
        self.retain_loss_w = 1.0
        self.ddim_eta = 0

        self.train_method = "text_encoder_full"   #choices: text_encoder_full', 'text_encoder_layer0', 'text_encoder_layer01', 'text_encoder_layer012', 'text_encoder_layer0123', 'text_encoder_layer01234', 'text_encoder_layer012345', 'text_encoder_layer0123456', 'text_encoder_layer01234567', 'text_encoder_layer012345678', 'text_encoder_layer0123456789', 'text_encoder_layer012345678910', 'text_encoder_layer01234567891011', 'text_encoder_layer0_11','text_encoder_layer01_1011', 'text_encoder_layer012_91011', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer
        self.norm_layer = False  # This is a flag; use True if you wish to update the norm layer.
        self.attack_method = "pgd"  # Choices: 'pgd', 'multi_pgd', 'fast_at', 'free_at'
        self.component = "all"     # Choices: 'all', 'ffn', 'attn'
        self.iterations = 1000
        self.save_interval = 200
        self.lr = 1e-5

        # Adversarial Attack Hyperparameters
        self.adv_prompt_num = 1
        self.attack_embd_type = "word_embd"  # Choices: 'word_embd', 'condition_embd'
        self.attack_type = "prefix_k"         # Choices: 'replace_k', 'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'
        self.attack_init = "latest"           # Choices: 'random', 'latest'
        self.attack_step = 30
        self.adv_prompt_update_step = 1
        self.attack_lr = 1e-3
        self.warmup_iter = 200

        # Override default values with any provided keyword arguments.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if self.retain_batch <= 0:
            raise ValueError("retain_batch should be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.image_size <= 0:
            raise ValueError("Image size should be a positive integer.")
        if self.iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Model config file {self.config_path} does not exist.")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if not os.path.exists(self.diffusers_config_path):
            raise FileNotFoundError(f"Diffusers config file {self.diffusers_config_path} does not exist.")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

adv_unlearn_config = AdvUnlearnConfig()

