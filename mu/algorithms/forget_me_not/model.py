# mu/algorithms/forget_me_not/model.py

import logging
from typing import List
import torch
import re
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, PretrainedConfig

from mu.core import BaseModel

class ForgetMeNotModel(BaseModel):
    """
    Model class for the Forget Me Not algorithm.
    Loads and prepares all necessary components from the Stable Diffusion model,
    applies TI weights if provided, and prepares the pipeline for attention training.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        devices = self.config.get('devices', ['cuda:0'])
        self.device = torch.device(devices[0] if devices and torch.cuda.is_available() else "cpu")

        self.tokenizer = None 
        self.vae = None 
        self.unet = None 
        self.text_encoder = None
        self.placeholder_token_ids = None

        placeholder_tokens = self.config.get('placeholder_tokens')
        initializer_tokens = self.config.get('initializer_tokens')
        placeholder_token_at_data = self.config.get('placeholder_token_at_data')
        

        if initializer_tokens is None:
            initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
        else:
            initializer_tokens = initializer_tokens.split("|")

        self.initializer_tokens = initializer_tokens
        class_token = "".join(initializer_tokens)
        self.class_token = class_token

        if placeholder_token_at_data is not None:
            tok, pat = placeholder_token_at_data.split("|")
            token_map = {tok: pat}

        else:
            token_map = {"DUMMY": "".join(placeholder_tokens)}

        self.placeholder_tokens = placeholder_tokens
        self.token_map = token_map

        ckpt_path = self.config.get('ckpt_path', '')
        pretrained_vae_name_or_path = self.config.get('pretrained_vae_name_or_path', '')
        placeholder_tokens = self.config.get('placeholder_tokens', []) 
        initializer_tokens = self.config.get('initializer_tokens', [])
        revision = self.config.get('revision', None)    
        type = self.config.get('type', 'train_ti')
        self.load_model(ckpt_path, pretrained_vae_name_or_path, self.placeholder_tokens, self.initializer_tokens, revision, type)

    def load_model(self, pretrained_model_name_or_path, pretrained_vae_name_or_path, placeholder_tokens :  List[str], initializer_tokens:  List[str], revision, type='ti', *args, **kwargs):
        """
        Load the model.
        """
        if not pretrained_model_name_or_path:
            raise ValueError("The `pretrained_model_name_or_path` is not provided.")
    
        if type == 'train_ti':
            tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=revision,
            )
        else: 
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=revision,
                use_fast=False,
            )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )

        placeholder_token_ids = []

        token_list = []
        for init_tok in initializer_tokens:
            token_ids = tokenizer.encode(init_tok)
            token_list = token_list + token_ids
        assert len(token_list) <= len(placeholder_tokens)

        placeholder_tokens = self.config.get('placeholder_tokens', '').split('|')

        for idx, token in enumerate(placeholder_tokens):
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different"
                    " `placeholder_token` that is not already in the tokenizer."
                )

            placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

            placeholder_token_ids.append(placeholder_token_id)

            # Load models and create wrapper for stable diffusion

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data

            if idx < len(token_list):
                token_embeds[placeholder_token_id] = token_embeds[token_list[idx]]
            else:
                init_tok = "<rand-1>"
                # <rand-"sigma">, e.g. <rand-0.5>
                sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

                token_embeds[placeholder_token_id] = (
                    torch.randn_like(token_embeds[0]) * sigma_val
                )
                print(
                    f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
                )
                print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        vae = AutoencoderKL.from_pretrained(
            pretrained_vae_name_or_path or pretrained_model_name_or_path,
            subfolder=None if pretrained_vae_name_or_path else "vae",
            revision=None if pretrained_vae_name_or_path else revision,
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
        )

        self.text_encoder = text_encoder.to(self.device)
        self.tokenizer = tokenizer
        self.vae = vae.to(self.device)
        self.unet = unet.to(self.device)
        self.placeholder_token_ids = placeholder_token_ids

        

    def save_model(self, output_path: str):
        """
        Save model weights after training.
        Uses a DiffusionPipeline for final saving of components.
        """
        pass