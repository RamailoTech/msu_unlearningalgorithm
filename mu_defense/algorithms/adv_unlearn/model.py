# mu_defense/algorithms/adv_unlearn/model.py

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from mu_defense.core import BaseModel 
from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_defense.algorithms.adv_unlearn import get_models_for_compvis, get_models_for_diffusers


class AdvUnlearnModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder_model_name_or_path = config.get("encoder_model_name_or_path")
        self.model_config_path = config.get("model_config_path")
        self.compvis_ckpt_path = config.get("compvis_ckpt_path")

        self.diffusers_model_name_or_path = config.get("diffusers_model_name_or_path")
        self.target_ckpt = config.get("target_ckpt")

        self.cache_path = config.get("cache_path")
        devices = config.get("devices")
        if isinstance(devices, str):
            self.devices = [f'cuda:{int(d.strip())}' for d in devices.split(',')]
        elif isinstance(devices, list):
            self.devices = devices
        else:
            raise ValueError("devices must be a comma-separated string or a list")
        
        self.backend = config.get("backend")

        self.load_model()

    def load_model(self):
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.encoder_model_name_or_path,
            subfolder="tokenizer",
            cache_dir=self.cache_path
        )
        # Load text encoder and wrap it
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.encoder_model_name_or_path,
            subfolder="text_encoder",
            cache_dir=self.cache_path
        ).to(self.devices[0])
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.devices[0])
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        # Load diffusion models
        if self.backend == "compvis":
            self.model_orig, self.sampler_orig, self.model, self.sampler = get_models_for_compvis(
                self.model_config_path,
                self.compvis_ckpt_path,
                self.devices
            )

        elif self.backend == "diffusers":
            self.model_orig, self.sampler_orig, self.model, self.sampler = get_models_for_diffusers(
                self.diffusers_model_name_or_path, self.devices, self.target_ckpt
            )


    def save_model(self, model: torch.nn.Module, output_path: str) -> None:
        """
        Save the model's state dictionary.

        Args:
            model (torch.nn.Module): The model to be saved.
            output_path (str): The file path where the model checkpoint will be stored.
        """
        if self.backend == "compvis":
            torch.save({"state_dict": model.state_dict()}, output_path)

        elif self.backend == "diffusers":
            model.save_pretrained(output_path)


    def apply_model(self, z: torch.Tensor, t: torch.Tensor, c):
        """
        Apply the diffusion model to produce an output.

        Args:
            z (torch.Tensor): Noisy latent vectors.
            t (torch.Tensor): Timestep tensor.
            c: Conditioning tensors.

        Returns:
            torch.Tensor: The output of the diffusion model.
        """
        return self.model.apply_model(z, t, c)
