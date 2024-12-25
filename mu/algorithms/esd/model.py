import torch
from typing import Any
from pathlib import Path

from mu.core import BaseModel
from mu.helpers import load_model_from_config

class ESDModel(BaseModel):
    """
    ESDModel handles loading, saving, and interacting with the Stable Diffusion model.
    """
    def __init__(self, model_config_path: str, ckpt_path: str, device: str, device_orig: str):
        """
        Initialize the ESDModel 
        
        Args:
            model_config_path (str): Path to the model configuration file.
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on (e.g., 'cuda:0').
            device_orig (str): Device to load the model on (e.g., 'cuda:0').

        """

        super().__init__()
        self.device = device
        self.device_orig = device_orig
        self.config_path = model_config_path
        self.ckpt_path = ckpt_path
        self.models = self.load_model(model_config_path, ckpt_path, device, device_orig)

    def load_model(self, model_config_path: str, ckpt_path: str, device: str, device_orig:str):
        """
        Load the Stable Diffusion model from config and checkpoint.

        Args:
            model_config_path (str): Path to the model configuration file.
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on.
            device_orig (str): Device to load the model on.


        Returns:
            [torch.nn.Module]: Loaded Stable Diffusion model.
        """
        model = load_model_from_config(model_config_path, ckpt_path, device)
        model_orig = load_model_from_config(model_config_path,ckpt_path, device_orig)

        return (model, model_orig)


    def save_model(self, model,output_path: str):
        """
        Save the trained model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        torch.save({"state_dict": model.state_dict()}, output_path)

    def get_learned_conditioning(self, prompts):
        """
        Obtain learned conditioning for given prompts.

        Args:
            prompts (list): List of prompt strings.

        Returns:
            Any: Learned conditioning tensors.
        """
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z, t, c):
        """
        Apply the model to generate outputs.

        Args:
            z (torch.Tensor): Noisy latent vectors.
            t (torch.Tensor): Timesteps.
            c (Any): Conditioning tensors.

        Returns:
            torch.Tensor: Model outputs.
        """
        model = self.models[0]
        return model.apply_model(z, t, c)
