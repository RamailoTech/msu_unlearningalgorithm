# erasediff_model.py

from algorithms.core.base_model import BaseModel
from stable_diffusion.ldm.util import instantiate_from_config
import torch
from typing import Any

class EraseDiffModel(BaseModel):
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

    def save_model(self, output_path: str):
        # Save the trained model
        torch.save({"state_dict": self.model.state_dict()}, output_path)

    def forward(self, input_data: Any) -> Any:
        # Implement the forward pass as needed
        pass

    def get_input(self, batch, key):
        return self.model.get_input(batch, key)

    def apply_model(self, x_noisy, t, cond):
        return self.model.apply_model(x_noisy, t, cond)

    def shared_step(self, batch):
        return self.model.shared_step(batch)
