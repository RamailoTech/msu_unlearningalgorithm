# mu/algorithms/selective_amnesia/model.py

import torch
from pathlib import Path
from typing import Any
import logging 

from stable_diffusion.ldm.util import instantiate_from_config

from mu.core import BaseModel
from mu.helpers import  load_config_from_yaml, rank_zero_print
from mu.algorithms.selective_amnesia.utils import modify_weights

class SelectiveAmnesiaModel(BaseModel):
    """
    Model class for Selective Amnesia.
    Loads the Stable Diffusion model and applies EWC constraints using the precomputed FIM.
    """

    def __init__(self, model_config_path: str, ckpt_path: str, device: str, opt_config, *args, **kwargs):
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
        self.config = load_config_from_yaml(model_config_path)
        self.ckpt_path = ckpt_path
        self.model = self.load_model(self.config, self.ckpt_path, opt_config)
        self.logger = logging.getLogger(__name__)

    def load_model(self, config, ckpt_path: str, opt_config):
        """
        Load the Stable Diffusion model from a configuration and checkpoint.

        Args:
            config: model config
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on.

        Returns:
            torch.nn.Module: The loaded Stable Diffusion model.
        """
        config.model.params.full_fisher_dict_pkl_path = opt_config.get('full_fisher_dict_pkl_path')

        model = instantiate_from_config(config.model)

        if ckpt_path : 
            rank_zero_print(f"Attempting to load state from {ckpt_path}")
            old_state = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in old_state:
                rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]

            #Check if we need to port weights from 4ch input to 8ch
            in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
            new_state = model.state_dict()
            in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
            in_shape = in_filters_current.shape
            if in_shape != in_filters_load.shape:
                rank_zero_print("Modifying weights to double number of input channels")
                keys_to_change = [
                    "model.diffusion_model.input_blocks.0.0.weight",
                    "model_ema.diffusion_modelinput_blocks00weight",
                ]
                scale = 1e-8
                for k in keys_to_change:
                    print("modifying input weights for compatibitlity")
                    old_state[k] = modify_weights(old_state[k], scale=scale, n=in_shape//4 - 1)

            m, u = model.load_state_dict(old_state, strict=False)
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)
        model.eval()   
        return model

    def save_model(self,model, output_path: str):
        """
        Save the trained model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        torch.save({"state_dict": model.state_dict()}, output_path)
