# semipermeable_membrane/model.py

import logging

import torch
from algorithms.semipermeable_membrane.src.models import model_util
from algorithms.semipermeable_membrane.src.models.spm import SPMLayer, SPMNetwork
from diffusers import StableDiffusionPipeline
from torch import nn


class SemipermeableMembraneModel(nn.Module):
    """
    SemipermeableMembraneModel loads the Stable Diffusion model and integrates SPMNetwork for concept editing.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Load precision
        self.weight_dtype = self._parse_precision(self.config.get('train', {}).get('precision', 'fp16'))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load stable diffusion models
        self.pipeline = self._load_pipeline()

        # Load SPM network
        self.network = SPMNetwork(
            unet=self.pipeline.unet,
            rank=self.config['network']['rank'],
            alpha=self.config['network']['alpha'],
            module=SPMLayer
        ).to(self.device, dtype=self.weight_dtype)

    def _load_pipeline(self):
        ckpt_path = self.config.get('pretrained_model', {}).get('ckpt_path', '')
        v2 = self.config.get('pretrained_model', {}).get('v2', False)
        v_pred = self.config.get('pretrained_model', {}).get('v_pred', False)

        self.logger.info(f"Loading pipeline from {ckpt_path}")
        # Load Stable Diffusion pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            ckpt_path,
            torch_dtype=self.weight_dtype
        ).to(self.device)
        pipeline.enable_attention_slicing()
        pipeline.unet.requires_grad_(False)
        pipeline.unet.eval()
        self.logger.info("Pipeline loaded successfully.")
        return pipeline

    def _parse_precision(self, precision_str: str):
        if precision_str == "fp16":
            return torch.float16
        elif precision_str == "bf16":
            return torch.bfloat16
        return torch.float32

    def forward(self, *args, **kwargs):
        # Implement forward if needed
        pass

    def save_model(self, output_path: str):
        self.logger.info(f"Saving model to {output_path}")
        # Save the SPM network weights
        self.network.save_weights(
            output_path,
            dtype=self.weight_dtype,
            metadata={
                "project": "semipermeable_membrane",
                "rank": self.config['network']['rank'],
                "alpha": self.config['network']['alpha']
            }
        )
        self.logger.info("Model saved successfully.")
