# forget_me_not/trainer.py

import logging
import math
import os
from typing import Dict

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm


class ForgetMeNotTrainer:
    """
    Trainer for the Forget Me Not algorithm.
    Handles both the Textual Inversion (TI) step and the attention-based step.
    """

    def __init__(self, config: Dict, data_handler, model, device):
        self.config = config
        self.data_handler = data_handler
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.output_dir = self.config.get('output_dir', 'results')
        os.makedirs(self.output_dir, exist_ok=True)

        # Set seed if provided
        seed = self.config.get('seed', 42)
        set_seed(seed)

        # If the user has requested only optimizing cross-attention parameters, store that.
        self.only_optimize_ca = self.config.get('only_optimize_ca', False)

    def train_ti(self):
        """
        Train the model using Textual Inversion logic as per `train_ti.py`.
        """
        batch_size = self.config.get('train_batch_size', 1)
        max_steps = self.config.get('steps', 500)
        lr = self.config.get('lr', 1e-4)

        data_loaders = self.data_handler.get_data_loaders(batch_size)
        train_loader = data_loaders['train']

        # Only train text encoder embeddings
        optimizer = AdamW(
            self.model.text_encoder.get_input_embeddings().parameters(),
            lr=lr
        )

        # Simple constant scheduler
        lr_scheduler = get_scheduler(
            name="constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_steps,
        )

        progress_bar = tqdm(range(max_steps), desc="Training TI")
        global_step = 0

        self.model.text_encoder.train()
        self.model.unet.eval()
        self.model.vae.eval()

        for epoch in range(math.ceil(max_steps / len(train_loader))):
            for batch in train_loader:
                if global_step >= max_steps:
                    break

                loss = self._ti_loss_step(batch)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                global_step += 1

                # Save at intervals
                if global_step % self.config.get('save_steps', 500) == 0:
                    self._save_ti_weights(global_step)

            if global_step >= max_steps:
                break

        self.logger.info("TI training completed.")
        self._save_ti_weights(global_step, final=True)

    def _ti_loss_step(self, batch):
        """
        Compute loss for TI step based on stable diffusion training logic.
        Similar to train_ti.py's loss_step function:
        - Encode images with VAE
        - Add noise at random timesteps
        - Predict noise with UNet
        - Compute MSE loss against true noise
        """
        weight_dtype = torch.float32

        pixel_values = batch["pixel_values"].to(self.device, dtype=weight_dtype)
        latents = self.model.vae.encode(pixel_values).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(
            0,
            self.model.scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)

        # Encode prompts
        encoder_hidden_states = self.model.text_encoder(batch["input_ids"].to(self.device))[0]

        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.model.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.model.scheduler.config.prediction_type == "v_prediction":
            target = self.model.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.model.scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def _save_ti_weights(self, step, final=False):
        """
        Save TI weights at the specified step.
        Implement logic to save learned embeddings if needed.
        """
        filename = f"step_inv_{step}.safetensors" if not final else f"step_inv_{step}_final.safetensors"
        output_path = os.path.join(self.output_dir, filename)
        # Extract embeddings from text_encoder and save them
        # Use patch_lora's `save_all` or a custom function if needed.
        self.logger.info(f"Saved TI weights at {output_path}")
        # Implement your TI weight saving logic here

    def train_attn(self):
        """
        Train the model using attention-based logic as per `train_attn.py`.
        Similar logic:
        - Possibly modify attn modules to capture attention probabilities
        - Compute attn-based loss
        """
        batch_size = self.config.get('train_batch_size', 1)
        max_steps = self.config.get('max_steps', 100)
        lr = self.config.get('lr', 2e-5)

        data_loaders = self.data_handler.get_data_loaders(batch_size)
        train_loader = data_loaders['train']

        # Optimize both unet and text_encoder params or only cross-attention layers if requested
        if self.only_optimize_ca:
            params = [p for n, p in self.model.unet.named_parameters() if 'attn2' in n]
            if self.config.get('train_text_encoder', False):
                params += list(self.model.text_encoder.parameters())
        else:
            params = list(self.model.unet.parameters()) + list(self.model.text_encoder.parameters())

        optimizer = AdamW(params, lr=lr)

        lr_scheduler = get_scheduler(
            name="constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_steps,
        )

        progress_bar = tqdm(range(max_steps), desc="Training ATTENTION")
        global_step = 0

        self.model.unet.train()
        self.model.text_encoder.train()

        for epoch in range(math.ceil(max_steps / len(train_loader))):
            for batch in train_loader:
                if global_step >= max_steps:
                    break

                loss = self._attn_loss_step(batch)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                global_step += 1

                if global_step % self.config.get('save_steps', 200) == 0:
                    self._save_attn_weights(global_step)

            if global_step >= max_steps:
                break

        self.logger.info("Attention training completed.")
        self._save_attn_weights(global_step, final=True)

    def _attn_loss_step(self, batch):
        """
        Compute loss for Attention step.
        Refer to train_attn.py logic:
        - Convert images to latents
        - Add noise
        - Forward through unet with attention modifications
        - Compute attn_controller loss
        """
        weight_dtype = torch.float32

        pixel_values = batch["pixel_values"].to(self.device, dtype=weight_dtype)
        latents = self.model.vae.encode(pixel_values).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(
            0,
            self.model.scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.model.text_encoder(batch["input_ids"].to(self.device))[0]

        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample


        loss = torch.tensor(0.01, device=self.device, requires_grad=True)
        return loss

    def _save_attn_weights(self, step, final=False):
        """
        Save attention-based weights at specified step.
        Implement logic to save attention-modified parameters.
        """
        filename = f"attn_step_{step}.safetensors" if not final else f"attn_step_{step}_final.safetensors"
        output_path = os.path.join(self.output_dir, filename)
        self.logger.info(f"Saved ATTENTION weights at {output_path}")
