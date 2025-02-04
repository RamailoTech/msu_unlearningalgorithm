# mu_attack/execs/adv_attack.py

import torch
import random
import wandb

from transformers import CLIPTextModel, CLIPTokenizer

from mu_attack.configs.adv_unlearn import AdvUnlearnConfig
from mu_attack.attackers.soft_prompt import SoftPromptAttack
from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_attack.helpers.utils import get_models


class AdvUnlearn:
    """
    Class for adversarial unlearning training.
    
    This class wraps the full training pipeline including adversarial attack 
    and model handling.
    """
    def __init__(self, config: AdvUnlearnConfig, **kwargs):
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        
        config.validate_config()

        self.prompt = config.prompt
        self.model_name_or_path = config.model_name_or_path
        self.cache_path = config.cache_path
        self.devices = [f'cuda:{int(d.strip())}' for d in config.devices.split(',')]
        self.attack_type = config.attack_type
        self.attack_embd_type = config.attack_embd_type
        self.attack_step = config.attack_step
        self.attack_lr = config.attack_lr
        self.attack_init = config.attack_init
        self.attack_init_embd = config.attack_init_embd
        self.attack_method = config.attack_method
        self.ddim_steps = config.ddim_steps
        self.ddim_eta = config.ddim_eta
        self.image_size = config.image_size
        self.adv_prompt_num = config.adv_prompt_num
        self.start_guidance = config.start_guidance
        self.config_path = config.config_path
        self.ckpt_path = config.ckpt_path
        self.criteria = torch.nn.MSELoss()

        # Initialize wandb
        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            reinit=True
        )

        # Load models
        self.load_models()

    def load_models(self):
        """Loads the tokenizer, text encoder, and models."""
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name_or_path, subfolder="tokenizer", cache_dir=self.cache_path
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name_or_path, subfolder="text_encoder", cache_dir=self.cache_path
        ).to(self.devices[0])
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.devices[0])
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        # Load base models
        self.model_orig, self.sampler_orig, self.model, self.sampler = get_models(
            self.config_path, self.ckpt_path, self.devices
        )

    def attack(self):
        """Performs the adversarial attack."""
        # Ensure words are in list format
        if isinstance(self.prompt, str):
            self.words = [self.prompt]
        elif isinstance(self.prompt, list):
            self.words = self.prompt
        else:
            raise ValueError("Prompt must be a string or a list of strings.")

        # Select a random word from the prompt list
        word = random.choice(self.words)

        # Get learned condition embeddings
        emb_0 = self.model_orig.get_learned_conditioning([''])
        emb_p = self.model_orig.get_learned_conditioning([word])

        # Initialize attack class
        sp_attack = SoftPromptAttack(
            model=self.model,
            model_orig=self.model_orig,
            tokenizer=self.tokenizer,
            text_encoder=self.custom_text_encoder,
            sampler=self.sampler,
            emb_0=emb_0,
            emb_p=emb_p,
            start_guidance=self.start_guidance,
            devices=self.devices,
            ddim_steps=self.ddim_steps,
            ddim_eta=self.ddim_eta,
            image_size=self.image_size,
            criteria=self.criteria,
            k=self.adv_prompt_num,
            all_embeddings=self.all_embeddings
        )


        self.adv_word_embd, self.adv_input_ids = sp_attack.attack(
            global_step=0,
            word=word,
            attack_round=0,
            attack_type=self.attack_type,
            attack_embd_type=self.attack_embd_type,
            attack_step=self.attack_step,
            attack_lr=self.attack_lr,
            attack_init=self.attack_init,
            attack_init_embd=self.attack_init_embd,
            attack_method=self.attack_method
        )


        return self.adv_word_embd, self.adv_input_ids


   