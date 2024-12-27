# mu/algorithms/saliency_unlearning/model.py

import torch
from pathlib import Path
from typing import Any, Dict

from mu.core import BaseModel
from mu.helpers import load_model_from_config


class SaliencyUnlearnModel(BaseModel):
    """
    SaliencyUnlearnModel handles loading, saving, and interacting with the Stable Diffusion model.
    Incorporates mask application for saliency-based unlearning.
    """

    def __init__(self, model_config_path: str, ckpt_path: str, mask: Dict[str, torch.Tensor], device: str):
        """
        Initialize the SaliencyUnlearnModel.

        Args:
            model_config_path (str): Path to the model configuration file.
            ckpt_path (str): Path to the model checkpoint.
            mask (Dict[str, torch.Tensor]): Mask dictionary for saliency.
            device (str): Device to load the model on (e.g., 'cuda:0').
        """
        super().__init__()
        self.device = device
        self.model_config_path = model_config_path
        self.ckpt_path = ckpt_path
        self.mask = mask
        self.model = self.load_model(model_config_path, ckpt_path, device)
        self.apply_mask()

    def load_model(self, model_config_path: str, ckpt_path: str, device: str):
        """
        Load the Stable Diffusion model from config and checkpoint.

        Args:
            model_config_path (str): Path to the model configuration file.
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on.

        Returns:
            torch.nn.Module: Loaded Stable Diffusion model.
        """

        return load_model_from_config(model_config_path, ckpt_path, device)


    def apply_mask(self):
        """
        Apply the mask to the model parameters to freeze or modify specific layers.
        This method ensures that only the intended parts of the model are trainable.
        """
        for name, param in self.model.named_parameters():
            if name in self.mask:
                param.requires_grad = False
                param.data *= self.mask[name].to(self.device)
                # Optionally, register hooks if dynamic masking is needed

    def save_model(self,model, output_path: str):
        """
        Save the trained model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        torch.save({"state_dict": model.state_dict()}, output_path)


    def forward(self, input_data: Any) -> Any:
        """
        Define the forward pass (if needed).

        Args:
            input_data (Any): Input data for the model.

        Returns:
            Any: Model output.
        """
        pass  # Implement if necessary

    def get_learned_conditioning(self, prompts: list) -> Any:
        """
        Obtain learned conditioning for given prompts.

        Args:
            prompts (list): List of prompt strings.

        Returns:
            Any: Learned conditioning tensors.
        """
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z: torch.Tensor, t: torch.Tensor, c: Any) -> torch.Tensor:
        """
        Apply the model to generate outputs.

        Args:
            z (torch.Tensor): Noisy latent vectors.
            t (torch.Tensor): Timesteps.
            c (Any): Conditioning tensors.

        Returns:
            torch.Tensor: Model outputs.
        """
        return self.model.apply_model(z, t, c)
