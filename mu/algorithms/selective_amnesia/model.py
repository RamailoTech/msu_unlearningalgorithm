import logging
from pathlib import Path
from typing import Any

import torch
from algorithms.selective_amnesia.utils import load_fim, modify_weights
from core.base_model import BaseModel
from omegaconf import OmegaConf

from stable_diffusion.ldm.util import instantiate_from_config

logger = logging.getLogger(__name__)

class SelectiveAmnesiaModel(BaseModel):
    """
    Model class for Selective Amnesia.
    Loads the Stable Diffusion model and applies EWC constraints using the precomputed FIM.
    """

    def __init__(self, config_path: str, ckpt_path: str, fim_path: str, device: str):
        super().__init__()
        self.device = device
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.fim_path = fim_path
        self.fim_dict = None
        self.model = self.load_model(config_path, ckpt_path, device)
        self.load_ewc_params()

    def load_model(self, config_path: str, ckpt_path: str, device: str):
        if isinstance(config_path, (str, Path)):
            config = OmegaConf.load(config_path)
        else:
            config = config_path

        old_state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in old_state:
            old_state = old_state["state_dict"]

        model = instantiate_from_config(config.model)
        # If input channels differ, modify weights as needed (example)
        in_filters_load = old_state.get("model.diffusion_model.input_blocks.0.0.weight", None)
        if in_filters_load is not None:
            curr_shape = model.state_dict()["model.diffusion_model.input_blocks.0.0.weight"].shape
            if in_filters_load.shape != curr_shape:
                logger.info("Modifying weights to double input channels...")
                old_state["model.diffusion_model.input_blocks.0.0.weight"] = modify_weights(
                    in_filters_load, scale=1e-8, n=(curr_shape[1]//in_filters_load.shape[1] - 1)
                )

        m,u = model.load_state_dict(old_state, strict=False)
        if len(m) > 0:
            logger.warning(f"Missing keys in state_dict load: {m}")
        if len(u) > 0:
            logger.warning(f"Unexpected keys in state_dict load: {u}")

        model.to(device)
        model.eval()
        return model

    def load_ewc_params(self):
        """
        Load EWC parameters from the precomputed FIM to guide forgetting training.
        """
        self.fim_dict = load_fim(self.fim_path)

    def save_model(self, output_path: str):
        torch.save({"state_dict": self.model.state_dict()}, output_path)

    def forward(self, *args, **kwargs):
        # Implement forward pass if needed
        pass

    def get_learned_conditioning(self, prompts: list) -> Any:
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z: torch.Tensor, t: torch.Tensor, c: Any) -> torch.Tensor:
        return self.model.apply_model(z, t, c)
