import logging
from mu.core.base_model import BaseModel
from mu.stable_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
from typing import Any
from pathlib import Path


class ESDModel(BaseModel):
    def __init__(self, config_path: str, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.model = self.load_model(config_path, ckpt_path, device)

    def load_model(self, config_path: str, ckpt_path: str, device: str):
        # Load model from config and checkpoint
        if isinstance(config_path, (str, Path)):
            config = OmegaConf.load(config_path)
        else:
            config = config_path  # If already a config object

        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
        model.cond_stage_model.device = device
        return model

    def forward(self, input_data: Any) -> Any:
        pass

    def get_learned_conditioning(self, prompts):
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z, t, c):
        return self.model.apply_model(z, t, c)
