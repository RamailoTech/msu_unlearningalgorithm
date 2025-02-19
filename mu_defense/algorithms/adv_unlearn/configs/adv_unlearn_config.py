# mu_defense/algorithms/adv_unlearn/configs/adv_unlearn_config.py

import os
from pathlib import Path
from mu_defense.core.base_config import BaseConfig

AVAILABLE_BACKENDS = ["compvis", "diffusers"]


class AdvUnlearnConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Inference & Model Paths
        self.model_config_path = (
            "configs/stable-diffusion/v1-inference.yaml"  # for compvis
        )
        self.compvis_ckpt_path = "models/sd-v1-4-full-ema.ckpt"
        self.encoder_model_name_or_path = "CompVis/stable-diffusion-v1-4"
        self.cache_path = ".cache"

        self.diffusers_model_name_or_path = ""
        self.target_ckpt = (
            None  # Optionally load a target checkpoint into model for diffuser sampling
        )

        # Devices & IO
        self.devices = "0,0"  # You can later parse this string into a list if needed.
        self.seperator = None
        self.use_sample = True

        self.output_dir = "outputs/adv_unlearn"

        # Image & Diffusion Sampling
        self.image_size = 512
        self.ddim_steps = 50
        self.start_guidance = 3.0
        self.negative_guidance = 1.0

        # Training Setup
        self.prompt = "nudity"
        self.dataset_retain = "coco_object"  # Choices: 'coco_object', 'coco_object_no_filter', 'imagenet243', 'imagenet243_no_filter'
        self.retain_batch = 5
        self.retain_train = "iter"  # Options: 'iter' or 'reg'
        self.retain_step = 1
        self.retain_loss_w = 1.0
        self.ddim_eta = 0

        self.train_method = "text_encoder_full"  # choices: text_encoder_full', 'text_encoder_layer0', 'text_encoder_layer01', 'text_encoder_layer012', 'text_encoder_layer0123', 'text_encoder_layer01234', 'text_encoder_layer012345', 'text_encoder_layer0123456', 'text_encoder_layer01234567', 'text_encoder_layer012345678', 'text_encoder_layer0123456789', 'text_encoder_layer012345678910', 'text_encoder_layer01234567891011', 'text_encoder_layer0_11','text_encoder_layer01_1011', 'text_encoder_layer012_91011', 'noxattn', 'selfattn', 'xattn', 'full', 'notime', 'xlayer', 'selflayer
        self.norm_layer = (
            False  # This is a flag; use True if you wish to update the norm layer.
        )
        self.attack_method = "pgd"  # Choices: 'pgd', 'multi_pgd', 'fast_at', 'free_at'
        self.component = "all"  # Choices: 'all', 'ffn', 'attn'
        self.iterations = 10
        self.save_interval = 200
        self.lr = 1e-5

        # Adversarial Attack Hyperparameters
        self.adv_prompt_num = 1
        self.attack_embd_type = "word_embd"  # Choices: 'word_embd', 'condition_embd'
        self.attack_type = "prefix_k"  # Choices: 'replace_k', 'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'
        self.attack_init = "latest"  # Choices: 'random', 'latest'
        self.attack_step = 30
        self.attack_init_embd = None
        self.adv_prompt_update_step = 1
        self.attack_lr = 1e-3
        self.warmup_iter = 200

        # backend
        self.backend = "compvis"

        experiment_name = f'AdvUnlearn-{self.prompt}-method_{self.train_method}_{self.component}-Attack_{self.attack_method}-Retain_{self.dataset_retain}_{self.retain_train}_{self.retain_loss_w}-lr_{self.lr}-AttackLr_{self.attack_lr}-{self.attack_type}_adv_num_{self.adv_prompt_num}-{self.attack_embd_type}-attack_init_{self.attack_init}-attack_step_{self.attack_step}-adv_update_{self.adv_prompt_update_step}-warmup_iter_{self.warmup_iter}'
    
        self.output_dir = Path("./outputs/results_with_retaining") / self.prompt / self.dataset_retain / self.attack_method /  f'AttackLr_{self.attack_lr}' / self.train_method / f'{self.component}' / self.attack_type / experiment_name



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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not self.backend in AVAILABLE_BACKENDS:
            raise ValueError(f"Backend should be in {AVAILABLE_BACKENDS}")


adv_unlearn_config = AdvUnlearnConfig()
