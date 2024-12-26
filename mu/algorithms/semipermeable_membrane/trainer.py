# semipermeable_membrane/trainer.py

import logging
from typing import List, Optional

import algorithms.semipermeable_membrane.src.engine.train_util as train_util
import torch
from algorithms.semipermeable_membrane.src.configs import config as config_pkg
from algorithms.semipermeable_membrane.src.configs import prompt as prompt_pkg
from algorithms.semipermeable_membrane.src.configs.config import RootConfig
from algorithms.semipermeable_membrane.src.configs.prompt import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
)
from algorithms.semipermeable_membrane.src.engine.sampling import sample
from algorithms.semipermeable_membrane.src.evaluation import eval_util
from algorithms.semipermeable_membrane.src.models import model_util
from torch.nn import MSELoss
from torch.optim import Adam


class SemipermeableMembraneTrainer:
    """
    Trainer for the Semipermeable Membrane algorithm.
    Handles the training loop and integrates model, data, and prompts.
    """

    def __init__(self, model, config, device, data_handler):
        self.model = model
        self.config = config
        self.device = device
        self.data_handler = data_handler
        self.logger = logging.getLogger("SemipermeableMembraneTrainer")

        # Load training parameters
        self.iterations = self.config.get("train", {}).get("iterations", 1000)
        self.lr = self.config.get("train", {}).get("lr", 1e-4)
        self.text_encoder_lr = self.config.get("train", {}).get("text_encoder_lr", 5e-5)
        self.unet_lr = self.config.get("train", {}).get("unet_lr", 1e-4)
        self.max_grad_norm = self.config.get("train", {}).get("max_grad_norm", 1.0)
        self.noise_scheduler_name = self.config.get("train", {}).get(
            "noise_scheduler", "ddim"
        )
        self.max_denoising_steps = self.config.get("train", {}).get(
            "max_denoising_steps", 50
        )

        # Initialize optimizer
        self.optimizer = Adam(
            [
                {
                    "params": self.model.pipeline.text_encoder.parameters(),
                    "lr": self.text_encoder_lr,
                },
                {"params": self.model.network.parameters(), "lr": self.unet_lr},
            ],
            lr=self.lr,
        )

        # Initialize scheduler if needed
        self.lr_scheduler = train_util.get_scheduler_fix(self.config, self.optimizer)

        # Define loss criterion
        self.criterion = MSELoss()

    def train(self):
        """
        Execute the training process.
        """
        add_prompts = self.config.get("add_prompts", False)
        guided_concepts = self.config.get("guided_concepts")
        preserve_concepts = self.config.get("preserve_concepts")

        # Prepare prompts using data handler
        old_texts, new_texts, retain_texts = self.data_handler.prepare_prompts(
            add_prompts=add_prompts,
            guided_concepts=guided_concepts,
            preserve_concepts=preserve_concepts,
        )

        # Initialize prompt embedding cache
        cache = PromptEmbedsCache()
        prompt_pairs: List[PromptEmbedsPair] = []

        with torch.no_grad():
            for target, positive, neutral, unconditional in self._load_prompts():
                # Encode prompts
                for prompt in [target, positive, neutral, unconditional]:
                    if cache[prompt] is None:
                        cache[prompt] = train_util.encode_prompts(
                            self.model.pipeline.tokenizer,
                            self.model.pipeline.text_encoder,
                            [prompt],
                        )

                # Create PromptEmbedsPair
                prompt_pair = PromptEmbedsPair(
                    criteria=self.criterion,
                    target=cache[target],
                    positive=cache[positive],
                    unconditional=cache[unconditional],
                    neutral=cache[neutral],
                    settings=None,  # Update if PromptSettings are used
                )

                prompt_pairs.append(prompt_pair)
                self.logger.info(f"Encoded prompt: {target}")

        self.logger.info(f"Total prompt pairs: {len(prompt_pairs)}")

        # Begin training loop
        for step in range(1, self.iterations + 1):
            self.optimizer.zero_grad()

            # Select a random prompt pair
            prompt_pair = prompt_pairs[torch.randint(0, len(prompt_pairs), (1,)).item()]

            # Sample timesteps
            timesteps_to = torch.randint(1, self.max_denoising_steps, (1,)).item()

            # Generate latents
            latents = train_util.get_initial_latents(
                noise_scheduler=self._get_noise_scheduler(),
                batch_size=prompt_pair.batch_size,
                height=prompt_pair.resolution,
                width=prompt_pair.resolution,
                num_channels=1,
            ).to(self.device, dtype=self.model.weight_dtype)

            # Forward pass through SPM network
            denoised_latents = self.model.network(
                latents=latents,
                embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                timesteps=timesteps_to,
                guidance_scale=3,
            )

            # Predict noise using unet
            noise_scheduler = self._get_noise_scheduler()
            noise_scheduler.set_timesteps(self.max_denoising_steps, device=self.device)
            denoised_latents = denoised_latents.to(self.device)
            target_latents = train_util.predict_noise(
                unet=self.model.pipeline.unet,
                noise_scheduler=noise_scheduler,
                timesteps_to=timesteps_to,
                denoised_latents=denoised_latents,
                embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            )

            # Compute loss
            loss = prompt_pair.loss(
                target_latents=target_latents,
                positive_latents=None,  # Update if positive latents are used
                neutral_latents=None,  # Update if neutral latents are used
                anchor_latents=None,  # Update if anchor latents are used
                anchor_latents_ori=None,
            )

            # Backward pass
            loss["loss"].backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.pipeline.text_encoder.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.network.parameters(), self.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Logging
            if step % 100 == 0 or step == 1:
                self.logger.info(
                    f"Step {step}/{self.iterations}: Loss={loss['loss'].item()}"
                )

            # Save checkpoints
            if step % self.config.get("save", {}).get("per_steps", 200) == 0:
                output_path = self.config.get("save", {}).get("path", "checkpoints/")
                output_name = f"{self.config.get('save', {}).get('name', 'spm_model')}_step{step}.safetensors"
                full_save_path = f"{output_path}/{output_name}"
                self.model.save_model(full_save_path)
                self.logger.info(f"Checkpoint saved at {full_save_path}")

            # Clean up
            del loss
            torch.cuda.empty_cache()

        # Save the final model
        final_model_path = f"{self.config.get('save', {}).get('path', 'checkpoints/')}/{self.config.get('save', {}).get('name', 'spm_model')}_final.safetensors"
        self.model.save_model(final_model_path)
        self.logger.info(
            f"Training completed. Final model saved at {final_model_path}."
        )

    def _get_noise_scheduler(self):
        """
        Initialize and return the noise scheduler based on the configuration.
        """
        return model_util.get_noise_scheduler(
            scheduler_name=self.noise_scheduler_name, device=self.device
        )

    def _load_prompts(self):
        """
        Load and yield prompts from the prompts file.
        """
        # Implement actual prompt loading logic from prompts_file
        prompts = prompt_pkg.load_prompts_from_yaml(self.data_handler.prompts_file)
        for settings in prompts:
            yield settings.target, settings.positive, settings.neutral, settings.unconditional
