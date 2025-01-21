# mu/algorithms/forget_me_not/trainer.py

import os
import math
import logging
from typing import Dict
from tqdm import tqdm
import torch
from torch.optim import AdamW
import itertools

from diffusers.optimization import get_scheduler
import torch.optim as optim
import torch.nn.functional as F
from diffusers import DDPMScheduler, DiffusionPipeline
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available

from lora_diffusion import (
    save_all
)

from mu.core import BaseTrainer
from mu.algorithms.forget_me_not.model import ForgetMeNotModel
from mu.algorithms.forget_me_not.utils import AttnController, MyCrossAttnProcessor
from mu.algorithms.forget_me_not.datasets.forget_me_not_dataset import ForgetMeNotDataset

class ForgetMeNotTrainer(BaseTrainer):
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

        self.setup_optimizer()

        self.noise_scheduler = DDPMScheduler.from_config(
            self.config.get('ckpt_path'), subfolder="scheduler"
        )

    def setup_optimizer(self, *args, **kwargs):
        """
        Setup optimizer for the model.
        """
        lr = float(self.config.get('lr'))
        scale_lr = self.config.get('scale_lr', False)
        gradient_accumulation_steps  = self.config.get('gradient_accumulation_steps', 1)
        train_batch_size = self.config.get('train_batch_size')

        if scale_lr: 
            ti_lr = lr * gradient_accumulation_steps * train_batch_size
        else:
            ti_lr = lr

        weight_decay_ti = float(self.config.get('weight_decay_ti', 0.1))
        self.ti_optimizer = optim.AdamW(
            self.model.text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

    def train_inversion(
        self,
        unet,
        text_encoder,
        dataloader,
        num_steps: int,
        index_no_updates,
        optimizer,
        save_steps: int,
        placeholder_token_ids,
        placeholder_tokens,
        save_path: str,
        lr_scheduler,
        accum_iter: int = 1,
        mixed_precision: bool = True,
        clip_ti_decay: bool = True,
    ):

        progress_bar = tqdm(range(num_steps))
        progress_bar.set_description("Steps")
        global_step = 0

        # Original Emb for TI
        orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

        index_updates = ~index_no_updates
        loss_sum = 0.0

        for epoch in range(math.ceil(num_steps / len(dataloader))):
            unet.eval()
            text_encoder.train()
            for batch in dataloader:

                lr_scheduler.step()

                with torch.set_grad_enabled(True):
                    loss = (
                        self._ti_loss_step(
                            batch,
                            mixed_precision=mixed_precision,
                        )
                        / accum_iter
                    )

                    loss.backward()
                    loss_sum += loss.detach().item()

                    if global_step % accum_iter == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                        with torch.no_grad():

                            # normalize embeddings
                            if clip_ti_decay:
                                pre_norm = (
                                    text_encoder.get_input_embeddings()
                                    .weight[index_updates, :]
                                    .norm(dim=-1, keepdim=True)
                                )

                                lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                                text_encoder.get_input_embeddings().weight[
                                    index_updates
                                ] = F.normalize(
                                    text_encoder.get_input_embeddings().weight[
                                        index_updates, :
                                    ],
                                    dim=-1,
                                ) * (
                                    pre_norm + lambda_ * (0.4 - pre_norm)
                                )
                                # print(pre_norm)

                            current_norm = (
                                text_encoder.get_input_embeddings()
                                .weight[index_updates, :]
                                .norm(dim=-1)
                            )

                            text_encoder.get_input_embeddings().weight[
                                index_no_updates
                            ] = orig_embeds_params[index_no_updates]


                    global_step += 1
                    progress_bar.update(1)

                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                if global_step % save_steps == 0:
                    save_all(
                        unet=unet,
                        text_encoder=text_encoder,
                        placeholder_token_ids=placeholder_token_ids,
                        placeholder_tokens=placeholder_tokens,
                        save_path=os.path.join(
                            save_path, f"step_inv_{global_step}.safetensors"
                        ),
                        save_lora=False,
                    )

                if global_step >= num_steps:
                    return
                
    def train_ti(self):
        """
        Train the model using Textual Inversion logic as per `train_ti.py`.
        """
        seed = self.config.get('seed', 42)
        gradient_checkpointing  = self.config.get('gradient_checkpointing', False)
        gradient_accumulation_steps  = self.config.get('gradient_accumulation_steps', 1)
        steps = self.config.get('steps')
        lr_warmup_steps = self.config.get('lr_warmup_steps', 0)
        train_batch_size = self.config.get('train_batch_size')

        torch.manual_seed(seed)

        if gradient_checkpointing:
            self.model.unet.enable_gradient_checkpointing()



        blur_amount = 20 
        tokenizer = self.model.tokenizer
        token_map = self.model.token_map

        train_dataloader = self.data_handler.load_data(tokenizer=tokenizer,blur_amount=blur_amount,token_map = token_map, train_batch_size=train_batch_size)

        index_no_updates = torch.arange(len(tokenizer)) != self.model.placeholder_token_ids[0]

        for tok_id in self.model.placeholder_token_ids:
            index_no_updates[tok_id] = False

        self.model.unet.requires_grad_(False)
        self.model.vae.requires_grad_(False)

        params_to_freeze = itertools.chain(
            self.model.text_encoder.text_model.encoder.parameters(),
            self.model.text_encoder.text_model.final_layer_norm.parameters(),
            self.model.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        for param in params_to_freeze:
            param.requires_grad = False

        lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=steps,
        )

        self.train_inversion(
            unet=self.model.unet,
            text_encoder=self.model.text_encoder,
            dataloader=train_dataloader,
            num_steps=steps,
            index_no_updates=index_no_updates,
            optimizer=self.ti_optimizer,
            save_steps=self.config.get('steps', 500),
            placeholder_token_ids=self.model.placeholder_token_ids,
            placeholder_tokens=self.model.placeholder_tokens,
            save_path=self.output_dir,
            lr_scheduler=lr_scheduler,
            accum_iter=gradient_accumulation_steps,
            mixed_precision=True,
            clip_ti_decay=True,
        )

        del self.ti_optimizer


    def _ti_loss_step(self, batch, t_mutliplier=1.0,mixed_precision=True,*args, **kwargs):
        """
        Compute loss for TI step based on stable diffusion training logic.
        Similar to train_ti.py's loss_step function:
        - Encode images with VAE
        - Add noise at random timesteps
        - Predict noise with UNet
        - Compute MSE loss against true noise
        """
        unet = self.model.unet
        vae = self.model.vae
        text_encoder = self.model.text_encoder
        scheduler = self.noise_scheduler
        weight_dtype = torch.float32

        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
        ).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(
            0,
            int(scheduler.config.num_train_timesteps * t_mutliplier),
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        if mixed_precision:
            with torch.cuda.amp.autocast():

                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(text_encoder.device)
                )[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        else:

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(text_encoder.device)
            )[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

        if batch.get("mask", None) is not None:
            mask = (
                batch["mask"]
                .to(model_pred.device)
                .reshape(
                    model_pred.shape[0], 1, batch["mask"].shape[2], batch["mask"].shape[3]
                )
            )
            # resize to match model_pred
            mask = (
                F.interpolate(
                    mask.float(),
                    size=model_pred.shape[-2:],
                    mode="nearest",
                )
                + 0.05
            )

            mask = mask / mask.mean()

            model_pred = model_pred * mask

            target = target * mask

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss


    def train_attn(self):
        """
        Train the model using attention-based logic as per `train_attn.py`.
        Similar logic:
        - Possibly modify attn modules to capture attention probabilities
        - Compute attn-based loss
        """
        use_ti = self.config.get('use_ti', False)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        mixed_precision = self.config.get('mixed_precision', "fp16")
        train_text_encoder = self.config.get('train_text_encoder', False)
        seed = self.config.get('seed', None)
        enable_xformers_memory_efficient_attention = self.config.get('enable_xformers_memory_efficient_attention', False)
        gradient_checkpointing = self.config.get('gradient_checkpointing', False)
        allow_tf32 = self.config.get('allow_tf32', False)
        scale_lr = self.config.get('scale_lr', False)
        train_batch_size = self.config.get('train_batch_size')
        use_8bit_adam = self.config.get('use_8bit_adam', False)
        adam_beta1 = float(self.config.get('adam_beta1', 0.9))
        adam_beta2 = float(self.config.get('adam_beta2', 0.999))
        adam_weight_decay = float(self.config.get('adam_weight_decay', 0.0))
        adam_epsilon = float(self.config.get('adam_epsilon', 1e-8))
        with_prior_preservation = self.config.get('with_prior_preservation', False)
        num_train_epochs = self.config.get('num_train_epochs', 1)
        lr_warmup_steps = self.config.get('lr_warmup_steps', 0)
        lr_num_cycles = self.config.get('lr_num_cycles', 1)
        lr_power = float(self.config.get('lr_power', 1.0))
        max_train_steps = self.config.get('max-steps', None)
        only_optimize_ca = self.config.get('only-xa')
        resume_from_checkpoint= self.config.get('resume_from_checkpoint', None)
        no_real_image = self.config.get('no_real_image', False) 
        max_grad_norm = self.config.get('max_grad_norm', 1.0)   
        checkpointing_steps = self.config.get('checkpointing_steps', 500)
        set_grads_to_none = self.config.get('set_grads_to_none', False)
        output_dir = self.config.get('output_dir', 'results')
        lr = float(self.config.get('lr'))
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

        if train_text_encoder and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )
        
        # self.logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_main_process:
            self.logger.info(accelerator.state)


        # If passed along, set the training seed now.
        if seed is not None:
            set_seed(seed)


        attn_controller = AttnController()
        module_count = 0
        for n, m in self.model.unet.named_modules():
            if n.endswith('attn2'):
                m.set_processor(MyCrossAttnProcessor(attn_controller, n))
                module_count += 1


        self.model.vae.requires_grad_(False)
        if not train_text_encoder:
            self.model.text_encoder.requires_grad_(False)

        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.model.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if gradient_checkpointing:
            self.model.unet.enable_gradient_checkpointing()
            if train_text_encoder:
                self.model.text_encoder.gradient_checkpointing_enable()

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if accelerator.unwrap_model(self.model.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {accelerator.unwrap_model(self.model.unet).dtype}. {low_precision_error_string}"
            )

        if train_text_encoder and accelerator.unwrap_model(self.model.text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(self.model.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if scale_lr:
            lr = (
                    lr * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        if only_optimize_ca:
            params_to_optimize = (
                itertools.chain(self.model.unet.parameters(), self.model.text_encoder.parameters()) if train_text_encoder
                else [p for n, p in self.model.unet.named_parameters()
                    if 'attn2' in n]
            )
            # print("only optimize cross attention...")
        else:
            params_to_optimize = (
                itertools.chain(self.model.unet.parameters(), self.model.text_encoder.parameters()) if train_text_encoder else self.model.unet.parameters()
            )
            # print("optimize unet...")
        optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = ForgetMeNotDataset(
            config = self.config,
            tokenizer=self.model.tokenizer,
            processed_dataset_dir=self.config.get('processed_dataset_dir'),
            dataset_type=self.config.get('dataset_type'),
            template_name=self.config.get('template_name'),
            template=self.config.get('template'),
            use_sample=self.config.get('use_sample'),
            size=self.config.get('size'),
            model= self.model,
            use_ti = use_ti
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: self._collate_fn(examples, with_prior_preservation),
            num_workers=0,
        )
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.config.get('lr_scheduler', 'linear'),
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
            num_cycles=lr_num_cycles,
            power=lr_power,
        )


        # Prepare everything with our `accelerator`.
        if train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.model.unet, self.model.text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.model.unet, optimizer, train_dataloader, lr_scheduler
            )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.model.vae.to(accelerator.device, dtype=weight_dtype)
        if not train_text_encoder:
            text_encoder = self.model.text_encoder.to(accelerator.device, dtype=weight_dtype)
        
        # text_encoder = self.model.text_encoder if train_text_encoder else None

        # Move text_encoder to the appropriate device
        # if text_encoder is not None:
        #     text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("forgetmenot")

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_dataset)}")
        self.logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        resume_step = 0

        debug_once = True
        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            if train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # show
                    if debug_once:
                        print(batch["instance_prompts"][0])
                        debug_once = False
                    # Convert images to latent space
                    latents = self.model.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * self.model.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if no_real_image:
                        noisy_latents = self.noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)
                    else:
                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # set concept_positions for this batch 
                    attn_controller.set_concept_positions(batch["concept_positions"])

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    ### collect attentions prob
                    loss = attn_controller.loss()
                    ###

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = params_to_optimize
                        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=set_grads_to_none)
                    attn_controller.zero_attn_probs()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            self.logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps:
                    break

            # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.config.get('ckpt_path'),
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=self.model.tokenizer,
                revision=self.config.get('revision'),
            )
            pipeline.save_pretrained(output_dir)

        accelerator.end_training()


    def _collate_fn(self,examples, with_prior_preservation=False):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        concept_positions = [example["concept_positions"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        instance_prompts = [example["instance_prompt"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)
        concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

        batch = {
            "instance_prompts": instance_prompts,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "concept_positions": concept_positions
        }
        return batch