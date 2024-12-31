# mu/algorithms/concept_ablation/model.py

import torch
from pathlib import Path
from typing import Any
import logging 

from mu.core import BaseModel
from mu.helpers import load_model_from_config


class ConceptAblationModel(BaseModel):
    """
    ConceptAblationModel handles loading, saving, and interacting with the Stable Diffusion model
    in the context of concept ablation.
    """

    def __init__(self, model_config_path: str, ckpt_path: str, device: str, *args, **kwargs):
        """
        Initialize the ConceptAblationModel.

        Args:
            model_config_path (str): Path to the model configuration file (YAML).
            ckpt_path (str): Path to the model checkpoint (CKPT).
            device (str): Device to load the model on (e.g., 'cuda:0').
        """
        super().__init__()
        self.device = device
        self.model_config_path = model_config_path
        self.ckpt_path = ckpt_path
        self.model = self.load_model(model_config_path, ckpt_path, device)
        self.logger = logging.getLogger(__name__)

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

    def save_model(self, output_path: str):
        """
        Save the trained model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        torch.save({"state_dict": self.model.state_dict()}, output_path)

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
