# mu/algorithms/semipermeable_membrane/trainer.py

import logging
import torch
from torch.nn import MSELoss
from tqdm import tqdm
from pathlib import Path


from mu.algorithms.semipermeable_membrane.model import SemipermeableMembraneModel
from mu.algorithms.semipermeable_membrane.data_handler import SemipermeableMembraneDataHandler
from mu.core import BaseTrainer
from mu.algorithms.semipermeable_membrane.src.models.spm import SPMNetwork
import mu.algorithms.semipermeable_membrane.src.engine.train_util as train_util
from mu.algorithms.semipermeable_membrane.src.configs.prompt import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
from mu.algorithms.semipermeable_membrane.src.engine.sampling import sample
from mu.algorithms.semipermeable_membrane.src.configs.config import RootConfig 


import gc


class SemipermeableMembraneTrainer(BaseTrainer):
    """
    Trainer for the Semipermeable Membrane algorithm.
    Handles the training loop and integrates model, data, and prompts.
    """

    def __init__(self, model: SemipermeableMembraneModel, config, data_handler: SemipermeableMembraneDataHandler,*args, **kwargs):
        super().__init__(model, config, **kwargs)
        self.network = model.network 
        self.model = model
        self.config = config
        self.device = model.device
        self.data_handler = data_handler
        self.logger = logging.getLogger('SemipermeableMembraneTrainer')
        self.setup_optimizer()
        self.verbose = getattr(self.config, "verbose", False)  # Default to False if not found

        config = RootConfig(**self.config)
        # Initialize scheduler if needed
        self.lr_scheduler = train_util.get_scheduler_fix(config, self.optimizer)

        # Define loss criterion
        self.criterion = MSELoss()
    
    

    def setup_optimizer(self,*args, **kwargs): 
        """
        Setup the optimizer
        """
        lr = self.config.get('train', {}).get('lr', 1e-4)
        text_encoder_lr = self.config.get('train', {}).get('text_encoder_lr', 5e-5)
        unet_lr = self.config.get('train', {}).get('unet_lr', 1e-4)

        self.trainable_params = self.network.prepare_optimizer_params(
            text_encoder_lr, unet_lr, lr
        )

        config = RootConfig(**self.config)

        optimizer_name, optimizer_args, self.optimizer = train_util.get_optimizer(
            config, self.trainable_params
        )

    @staticmethod
    def flush_cache():
        """
        Flush the cache.
        """
        torch.cuda.empty_cache()
        gc.collect()
    
    def train(self, *args, **kwargs):
        """
        Execute the training process.
        """
        iterations = self.config.get('train', {}).get('iterations', 1000)
        max_denoising_steps = self.config.get('train', {}).get('max_denoising_steps', 1000)
        verbose = self.config.get('verbose')
        max_grad_norm = self.config.get('train', {}).get('max_grad_norm', 0)
        prompts = self.data_handler.load_data()
        rank = self.config.get('network', {}).get('rank', 1)
        alpha = self.config.get('network', {}).get('alpha', 1.0)
        save_per_steps = self.config.get('save', {}).get('per_steps', 1000)
        train_iterations = self.config.get('train', {}).get('iterations', 1000)

        self.model.model_metadata =  {
            "prompts": ",".join([prompt.target for prompt in prompts]),
            "rank": str(rank),
            "alpha": str(alpha),
        }


        cache = PromptEmbedsCache()
        prompt_pairs: list[PromptEmbedsPair] = []

        with torch.no_grad():
            for settings in prompts:
                for prompt in [
                    settings.target,
                    settings.positive,
                    settings.neutral,
                    settings.unconditional,
                ]:
                    if cache[prompt] == None:
                        cache[prompt] = train_util.encode_prompts(
                            self.model.tokenizer, self.model.text_encoder , [prompt]
                        )

                prompt_pair = PromptEmbedsPair(
                    self.criterion,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
                assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
                prompt_pairs.append(prompt_pair)
                print(f"norm of target: {prompt_pair.target.norm()}")

        self.logger.info(f"Total prompt pairs: {len(prompt_pairs)}")

        SemipermeableMembraneTrainer.flush_cache()

        pbar = tqdm(range(iterations))
        loss = None

        for i in pbar:
            with torch.no_grad():
                self.model.noise_scheduler.set_timesteps(
                    max_denoising_steps, device=self.device
                )

                self.optimizer.zero_grad()

                prompt_pair: PromptEmbedsPair = prompt_pairs[
                    torch.randint(0, len(prompt_pairs), (1,)).item()
                ]

                timesteps_to = torch.randint(
                    1, max_denoising_steps, (1,)
                ).item()

                height, width = (
                    prompt_pair.resolution,
                    prompt_pair.resolution,
                )
                if prompt_pair.dynamic_resolution:
                    height, width = train_util.get_random_resolution_in_bucket(
                        prompt_pair.resolution
                    )

                if verbose:
                    print("guidance_scale:", prompt_pair.guidance_scale)
                    print("resolution:", prompt_pair.resolution)
                    print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                    if prompt_pair.dynamic_resolution:
                        print("bucketed resolution:", (height, width))
                    print("batch_size:", prompt_pair.batch_size)

                latents = train_util.get_initial_latents(
                    self.model.noise_scheduler, prompt_pair.batch_size, height, width, 1
                ).to(self.device, dtype=self.model.weight_dtype)

                with self.network:
                    denoised_latents = train_util.diffusion(
                        self.model.unet,
                        self.model.noise_scheduler,
                        latents,
                        train_util.concat_embeddings(
                            prompt_pair.unconditional,
                            prompt_pair.target,
                            prompt_pair.batch_size,
                        ),
                        start_timesteps=0,
                        total_timesteps=timesteps_to,
                        guidance_scale=3,
                    )

                self.model.noise_scheduler.set_timesteps(1000)

                current_timestep = self.model.noise_scheduler.timesteps[
                    int(timesteps_to * 1000 / max_denoising_steps)
                ]

                positive_latents = train_util.predict_noise(
                    self.model.unet,
                    self.model.noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    train_util.concat_embeddings(
                        prompt_pair.unconditional,
                        prompt_pair.positive,
                        prompt_pair.batch_size,
                    ),
                    guidance_scale=1,
                ).to("cpu", dtype=torch.float32)
                neutral_latents = train_util.predict_noise(
                    self.model.unet,
                    self.model.noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    train_util.concat_embeddings(
                        prompt_pair.unconditional,
                        prompt_pair.neutral,
                        prompt_pair.batch_size,
                    ),
                    guidance_scale=1,
                ).to("cpu", dtype=torch.float32)

            with self.network:
                target_latents = train_util.predict_noise(
                    self.model.unet,
                    self.model.noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    train_util.concat_embeddings(
                        prompt_pair.unconditional,
                        prompt_pair.target,
                        prompt_pair.batch_size,
                    ),
                    guidance_scale=1,
                ).to("cpu", dtype=torch.float32)

            # ------------------------- latent anchoring part -----------------------------

            if prompt_pair.action == "erase_with_la":
                # noise sampling
                anchors = sample(prompt_pair, tokenizer=self.model.tokenizer, text_encoder=self.model.text_encoder)

                # get latents
                repeat = prompt_pair.sampling_batch_size // prompt_pair.batch_size
                with self.network:
                    anchor_latents = train_util.predict_noise(
                        self.model.unet,
                        self.model.noise_scheduler,
                        current_timestep,
                        denoised_latents.repeat(repeat, 1, 1, 1),
                        anchors,
                        guidance_scale=1,
                    ).to("cpu", dtype=torch.float32)

                with torch.no_grad():
                    anchor_latents_ori = train_util.predict_noise(
                        self.model.unet,
                        self.model.noise_scheduler,
                        current_timestep,
                        denoised_latents.repeat(repeat, 1, 1, 1),
                        anchors,
                        guidance_scale=1,
                    ).to("cpu", dtype=torch.float32)
                anchor_latents_ori.requires_grad_ = False

            else:
                anchor_latents = None
                anchor_latents_ori = None 

            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False

            loss = prompt_pair.loss(
                target_latents=target_latents,
                positive_latents=positive_latents,
                neutral_latents=neutral_latents,
                anchor_latents=anchor_latents,
                anchor_latents_ori=anchor_latents_ori,
            )

            loss["loss"].backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_params, max_grad_norm, norm_type=2
                )
            self.optimizer.step()
            self.lr_scheduler.step()

            pbar.set_description(f"Loss*1k: {loss['loss'].item()*1000:.4f}")

                    # save model
            if (
                i % save_per_steps == 0
                and i != 0
                and i != train_iterations - 1
            ):
                self.logger.info("Saving...")
                # Create output directory if it doesn't exist
                output_dir = Path(getattr(self.config, "output_dir", "./outputs"))

                output_dir.mkdir(parents=True, exist_ok=True)

                output_name = output_dir / f"semipermeable_membrane_{self.config.get('template_name')}_{i}_steps.safetensors"

                self.model.save_model(
                    self.network,
                    output_name,
                    dtype=self.model.save_weight_dtype,
                    metadata=self.model.model_metadata,
                )
        
            del (
                positive_latents,
                neutral_latents,
                target_latents,
                latents,
                anchor_latents,
                anchor_latents_ori,
            )
            
            SemipermeableMembraneTrainer.flush_cache()


        self.logger.info("Saving...")

        # Create output directory if it doesn't exist
        output_dir = Path(self.config.get("output_dir", "./outputs"))

        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = output_dir / f"semipermeable_membrane_{self.config.get('template_name')}_last.safetensors"


        self.model.save_model(
            self.network,
            output_name,
            dtype=self.model.save_weight_dtype,
            metadata=self.model.model_metadata,
        )

        del (
            self.model.unet,
            self.model.noise_scheduler,
            loss,
            self.optimizer,
            self.network,
        )

        SemipermeableMembraneTrainer.flush_cache()



