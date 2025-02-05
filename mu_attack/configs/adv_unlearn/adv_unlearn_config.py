import os
from pathlib import Path
from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent

class AdvUnlearnConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Inference & Model Paths
        self.config_path = current_dir / "model_config.yaml"
        self.ckpt_path = "models/sd-v1-4-full-ema.ckpt"
        self.model_name_or_path = "CompVis/stable-diffusion-v1-4"
        self.target_ckpt = None
        self.diffusers_model_name_or_path = "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50"
        
        # Devices & IO
        self.devices = "0,0"  # You can later parse this string into a list if needed.
        self.seperator = None
        self.cache_path = ".cache"
        
        # Image & Diffusion Sampling
        self.start_guidance = 3.0
        self.ddim_steps = 50


        # Training Setup
        self.image_size = 512
        self.prompt = "nudity"
        self.attack_method = "pgd"  # Choices: 'pgd', 'multi_pgd', 'fast_at', 'free_at'
        self.ddim_eta = 0

        # Adversarial Attack Hyperparameters
        self.adv_prompt_num = 1
        self.attack_init_embd = None
        self.attack_embd_type = "word_embd"  # Choices: 'word_embd', 'condition_embd'
        self.attack_type = "prefix_k"         # Choices: 'replace_k', 'add', 'prefix_k', 'suffix_k', 'mid_k', 'insert_k', 'per_k_words'
        self.attack_init = "latest"           # Choices: 'random', 'latest'
        self.attack_step = 30
        self.attack_lr = 1e-3

        #backend
        self.backend = "diffusers"

        #wandb configs
        self.project_name = "quick-canvas-machine-unlearning"
        self.experiment_name = f'AdvUnlearn-{self.prompt}-method_Attack_{self.attack_method}'
    

        # Override default values with any provided keyword arguments.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Model config file {self.config_path} does not exist.")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")

adv_unlearn_config = AdvUnlearnConfig()

