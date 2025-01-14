# mu/algorithms/concept_ablation/model.py

import torch
from pathlib import Path
from typing import Any
import logging 

from stable_diffusion.ldm.util import instantiate_from_config

from mu.core import BaseModel
from mu.helpers import  load_config_from_yaml



class ConceptAblationModel(BaseModel):
    """
    ConceptAblationModel handles loading, saving, and interacting with the Stable Diffusion model
    in the context of concept ablation.
    """

    def __init__(self, train_config, model_config_path: str, ckpt_path: str, device: str, *args, **kwargs):
        """
        Initialize the ConceptAblationModel.

        Args:
            model_config_path (str): Path to the model configuration file (YAML).
            ckpt_path (str): Path to the model checkpoint (CKPT).
            device (str): Device to load the model on (e.g., 'cuda:0').
        """
        super().__init__()
        self.device = device
        self.train_config = train_config
        self.model_config_path = model_config_path
        self.config = load_config_from_yaml(model_config_path)
        self.ckpt_path = ckpt_path
        self.model = self.load_model(self.train_config,self.config, self.ckpt_path, device)
        self.logger = logging.getLogger(__name__)

    def load_model(self, train_config, config, ckpt_path: str, device: str):
        """
        Load the Stable Diffusion model from a configuration and checkpoint.

        Args:
            config: model config
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on.

        Returns:
            torch.nn.Module: The loaded Stable Diffusion model.
        """
        dataset_type = train_config.get('dataset_type')
        base_lr = train_config.get('base_lr')
        modifier_token = train_config.get('modifier_token')
        freeze_model = train_config.get('freeze_model')
        loss_type_reverse = train_config.get('loss_type_reverse')

        if dataset_type in ["unlearncanvas", "i2p"]:
            config.model.params.cond_stage_trainable = False
            config.model.params.freeze_model = "crossattn-kv"

        if base_lr is not None:
            config.model.params.base_lr = base_lr

        if modifier_token is not None:
            config.model.params.cond_stage_config.params.modifier_token = modifier_token
        
        if ckpt_path : 
            config.model.params.ckpt_path = None

        if freeze_model is not None:
            config.model.params.freeze_model = freeze_model

        config.model.params.loss_type_reverse = loss_type_reverse

        model = instantiate_from_config(config.model)
        if ckpt_path : 
            st = torch.load(ckpt_path, map_location='cpu')["state_dict"]
            token_weights = st["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
            model.load_state_dict(st, strict=False)
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[
                :token_weights.shape[0]] = token_weights
            
        return model

    def save_model(self,model, output_path: str):
        """
        Save the trained model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        torch.save({"state_dict": model.state_dict()}, output_path)
