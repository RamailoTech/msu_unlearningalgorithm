from core.base_model import BaseModel
from mu.stable_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
from pathlib import Path
from typing import Any


class ConceptAblationModel(BaseModel):
    """
    ConceptAblationModel handles loading, saving, and interacting with the Stable Diffusion model
    in the context of concept ablation.
    """

    def __init__(self, config_path: str, ckpt_path: str, device: str):
        """
        Initialize the ConceptAblationModel.

        Args:
            config_path (str): Path to the model configuration file (YAML).
            ckpt_path (str): Path to the model checkpoint (CKPT).
            device (str): Device to load the model on (e.g., 'cuda:0').
        """
        super().__init__()
        self.device = device
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.model = self.load_model(config_path, ckpt_path, device)

    def load_model(self, config_path: str, ckpt_path: str, device: str):
        """
        Load the Stable Diffusion model from a configuration and checkpoint.

        Args:
            config_path (str): Path to the model configuration file.
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on.

        Returns:
            torch.nn.Module: The loaded Stable Diffusion model.
        """
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
        """
        Define the forward pass (if needed for integration with certain pipelines).

        Args:
            input_data (Any): Input data for the model.

        Returns:
            Any: Model output (if needed).
        """
        # Typically, Stable Diffusion forward operations are handled differently (via apply_model).
        # Implement if your training pipeline requires a direct forward call.
        pass

    def get_learned_conditioning(self, prompts: list) -> Any:
        """
        Obtain learned conditioning for given prompts.

        Args:
            prompts (list): List of prompt strings.

        Returns:
            Any: Learned conditioning tensors for the model.
        """
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z: torch.Tensor, t: torch.Tensor, c: Any) -> torch.Tensor:
        """
        Apply the model to generate outputs from noisy latent vectors.

        Args:
            z (torch.Tensor): Noisy latent vectors.
            t (torch.Tensor): Timesteps.
            c (Any): Conditioning information.

        Returns:
            torch.Tensor: Model outputs (denoised latents).
        """
        return self.model.apply_model(z, t, c)
