# mu_attack/execs/adv_attack.py

import torch
import wandb

from mu_attack.configs.adv_unlearn import AdvAttackConfig
from mu_attack.attackers.soft_prompt import SoftPromptAttack
from mu_attack.helpers.utils import get_models_for_compvis, get_models_for_diffusers


class AdvAttack:
    def __init__(self, config: AdvAttackConfig):
        self.config = config.__dict__
        # Do not set self.prompt from the config; remove the dependency.
        self.encoder_model_name_or_path = config.encoder_model_name_or_path
        self.cache_path = config.cache_path
        self.devices = [f"cuda:{int(d.strip())}" for d in config.devices.split(",")]
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

        # Initialize wandb (if needed)
        wandb.init(
            project=config.project_name, name=config.experiment_name, reinit=True
        )

        # self.load_models()

    def load_models(self):
        if self.backend == "compvis":
            self.model_orig, self.sampler_orig, self.model, self.sampler = (
                get_models_for_compvis(
                    self.config_path, self.compvis_ckpt_path, self.devices
                )
            )
        elif self.backend == "diffusers":
            self.model_orig, self.sampler_orig, self.model, self.sampler = (
                get_models_for_diffusers(
                    self.diffusers_model_name_or_path, self.target_ckpt, self.devices
                )
            )

    def attack(self, word, global_step, attack_round):
        """
        Perform the adversarial attack using the given word.

        Args:
            word (str): The current prompt to attack.
            global_step (int): The current global training step.
            attack_round (int): The current attack round.

        Returns:
            tuple: (adversarial embedding, input_ids)
        """
        # Now, use the passed `word` for the attack instead of self.prompt.
        # (Everything else in this method remains the same.)
        sp_attack = SoftPromptAttack(
            model=self.model,
            model_orig=self.model_orig,
            tokenizer=self.tokenizer,
            text_encoder=self.custom_text_encoder,
            sampler=self.sampler,
            emb_0=self._get_emb_0(),
            emb_p=self._get_emb_p(word),
            start_guidance=self.start_guidance,
            devices=self.devices,
            ddim_steps=self.ddim_steps,
            ddim_eta=self.ddim_eta,
            image_size=self.image_size,
            criteria=self.criteria,
            k=self.adv_prompt_num,
            all_embeddings=self.all_embeddings,
            backend=self.backend,
        )
        return sp_attack.attack(
            global_step,
            word,
            attack_round,
            self.attack_type,
            self.attack_embd_type,
            self.attack_step,
            self.attack_lr,
            self.attack_init,
            self.attack_init_embd,
            self.attack_method,
        )

    # Example helper methods to get embeddings from model_orig.
    def _get_emb_0(self):
        if self.backend == "compvis":
            return self.model_orig.get_learned_conditioning([""])
        else:
            # For diffusers, you need to define your own method (e.g., using self.encode_text(""))
            return self.encode_text("")

    def _get_emb_p(self, word):
        if self.backend == "compvis":
            return self.model_orig.get_learned_conditioning([word])
        else:
            return self.encode_text(word)

    def encode_text(self, text):
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.devices[0])
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        return text_embeddings
