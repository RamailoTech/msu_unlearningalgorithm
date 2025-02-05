# mu_attack/execs/adv_attack.py

import torch
import random
import wandb

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from mu_attack.configs.adv_unlearn import AdvUnlearnConfig
from mu_attack.attackers.soft_prompt import SoftPromptAttack
from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_attack.helpers.utils import get_models_for_compvis, get_models_for_diffusers


class AdvAttack:
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
        self.encoder_model_name_or_path = config.encoder_model_name_or_path
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
        self.compvis_ckpt_path = config.compvis_ckpt_path
        self.backend = config.backend
        self.diffusers_model_name_or_path = config.diffusers_model_name_or_path
        self.target_ckpt = config.target_ckpt
        self.criteria = torch.nn.MSELoss()

        # Initialize wandb
        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            reinit=True
        )

        # Load models
        self.load_models()

    def encode_text(self, text):
        """Encodes text into a latent space using CLIP from Diffusers."""
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.devices[0])  # Move to correct device

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]  # Take the first output (hidden states)

        return text_embeddings

    def load_models(self):
        """Loads the tokenizer, text encoder, and models."""
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.encoder_model_name_or_path, subfolder="tokenizer", cache_dir=self.cache_path
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.encoder_model_name_or_path, subfolder="text_encoder", cache_dir=self.cache_path
        ).to(self.devices[0])
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.devices[0])
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        # Load base models
        if self.backend == "compvis":
            self.model_orig, self.sampler_orig, self.model, self.sampler = get_models_for_compvis(
                self.config_path, self.compvis_ckpt_path, self.devices
            )
        elif self.backend == "diffusers":
            self.model_orig, self.sampler_orig, self.model, self.sampler = get_models_for_diffusers(
                self.diffusers_model_name_or_path, self.target_ckpt, self.devices
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

        if self.backend == "compvis":
            # CompVis uses `get_learned_conditioning`
            emb_0 = self.model_orig.get_learned_conditioning([''])
            emb_p = self.model_orig.get_learned_conditioning([word])
        elif self.backend == "diffusers":
            # Diffusers requires explicit encoding via CLIP
            emb_0 = self.encode_text("")
            emb_p = self.encode_text(word)

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
            all_embeddings=self.all_embeddings,
            backend = self.backend
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


   