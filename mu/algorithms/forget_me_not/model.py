# forget_me_not/model.py

import logging

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion.patch_lora import (
    apply_learned_embed_in_clip,
    parse_safeloras_embeds,
    safe_open,
)


class ForgetMeNotModel:
    """
    Model class for the Forget Me Not algorithm.
    Loads and prepares all necessary components from the Stable Diffusion model,
    applies TI weights if provided, and prepares the pipeline for attention training.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pretrained_model_path = self.config.get("pretrained_model_name_or_path", "")
        self.ti_weight_path = self.config.get("ti_weight_path", None)

        # Load tokenizer, text encoder, UNet, and VAE
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        ).to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path, subfolder="unet"
        ).to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        )

        # If TI weights are provided, apply them
        if self.ti_weight_path:
            self._load_ti_weights(self.ti_weight_path)

    def _load_ti_weights(self, ti_weight_path: str):
        """
        Load and apply Textual Inversion (TI) weights from the given path.
        Uses logic from train_ti.py and patch_lora.py utilities.
        """
        self.logger.info(f"Loading TI weights from {ti_weight_path}")
        safeloras = safe_open(ti_weight_path, framework="pt", device="cpu")
        tok_dict = parse_safeloras_embeds(safeloras)

        for token, embed in tok_dict.items():
            apply_learned_embed_in_clip(
                {token: embed},
                self.text_encoder,
                self.tokenizer,
                token=token,
                idempotent=True,
            )

    def save_model(self, output_path: str):
        """
        Save model weights after training.
        Uses a DiffusionPipeline for final saving of components.
        """
        self.logger.info(f"Saving model to {output_path}")
        pipeline = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        pipeline.save_pretrained(output_path)
        self.logger.info("Model saved successfully.")
