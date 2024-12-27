#  mu/algorithms/semipermeable_membrane/model.py

import logging
import torch
from torch import nn

from mu.core import BaseModel
from mu.algorithms.semipermeable_membrane.src.models.spm import SPMNetwork, SPMLayer
from mu.algorithms.semipermeable_membrane.src.models.model_util import load_models


class SemipermeableMembraneModel(BaseModel):
    """
    SemipermeableMembraneModel loads the Stable Diffusion model and integrates SPMNetwork for concept editing.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        # Load precision
        self.weight_dtype = self._parse_precision(self.config.get('train', {}).get('precision', 'fp16'))
        self.save_weight_dtype = self._parse_precision(self.config.get('train', {}).get('precision', 'fp16'))
        devices = self.config.get('devices', ['cuda:0'])
        self.device = torch.device(devices[0] if devices and torch.cuda.is_available() else "cpu")
        self.network = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None
        self.unet = None
        self.model_metadata = None
        self.load_model()
        

    def load_model(self,*args, **kwargs):
        """
        Load the model        
        """

        ckpt_path = self.config.get('pretrained_model', {}).get('ckpt_path', '')
        v2 = self.config.get('pretrained_model', {}).get('v2', False)
        v_pred = self.config.get('pretrained_model', {}).get('v_pred', False)

        scheduler_name = self.config.get('train', {}).get('noise_scheduler', 'linear')

        (
        tokenizer, 
        text_encoder, 
        unet, 
        noise_scheduler, 
        pipe
        ) = load_models(
            ckpt_path,
            scheduler_name=scheduler_name,
            v2=v2,
            v_pred=v_pred,
        )

        text_encoder.to(self.device, dtype=self.weight_dtype)
        text_encoder.eval()

        unet.to(self.device, dtype=self.weight_dtype)
        unet.enable_xformers_memory_efficient_attention()
        unet.requires_grad_(False)
        unet.eval()

        rank = self.config.get('network', {}).get('rank', 1)
        alpha = self.config.get('network', {}).get('alpha', 1.0)
        network = SPMNetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            module=SPMLayer,
        ).to(self.device, dtype=self.weight_dtype)

        self.network = network
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.unet = unet



    def _parse_precision(self, precision_str: str):
        if precision_str == "fp16":
            return torch.float16
        elif precision_str == "bf16":
            return torch.bfloat16
        return torch.float32


    def save_model(self, model, output_path: str, dtype, metadata, *args, **kwargs):
        """
        Save the model weights to the output path
        """
        #TODO
        self.logger.info(f"Saving model to {output_path}")
        # Save the SPM network weights
        model.save_weights(
            output_path,
            dtype=dtype,
            metadata=metadata
        )
        self.logger.info("Model saved successfully.")
