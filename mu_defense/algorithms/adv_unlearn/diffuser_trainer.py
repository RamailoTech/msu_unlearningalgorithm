# mu_defense/algorithms/adv_unlearn/diffuser_trainer.py

import torch
import random
import wandb
import logging

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.nn import MSELoss
from diffusers import StableDiffusionPipeline

from mu.core import BaseTrainer
from mu_defense.algorithms.adv_unlearn.utils import (
    id2embedding,
    param_choices,
    get_train_loss_retain_diffuser,
    save_text_encoder,
    save_history,
    sample_model_for_diffuser,
)


from mu_attack.execs.adv_attack import AdvAttack
from mu_attack.configs.adv_unlearn import AdvAttackConfig
from mu_defense.algorithms.adv_unlearn import AdvUnlearnDatasetHandler


class AdvUnlearnDiffuserTrainer(BaseTrainer):
    """
    Trainer for adversarial unlearning.

    This trainer performs the adversarial prompt update and retention-based
    regularized training loop for Diffusers models.
    """
    def __init__(self, model, config: dict, devices: list, **kwargs):
        """
        Initialize the AdvUnlearnCompvisTrainer.
        """
        super().__init__(model, config, **kwargs)
        self.devices = devices

        # Unpack models and samplers from the provided model loader.
        self.model = model.model            # trainable diffusion model
        self.model_orig = model.model_orig  # frozen diffusion model (set to eval)
        self.sampler = model.sampler
        self.sampler_orig = model.sampler_orig
        self.model_loader = model 
        self.text_encoder = model.text_encoder
        self.vae = model.vae
        self.safety_checker = model.safety_checker
        self.feature_extractor = model.feature_extractor

        # Other loaded components.
        self.tokenizer = model.tokenizer
        self.custom_text_encoder = model.custom_text_encoder
        self.all_embeddings = model.all_embeddings

        # Loss criterion.
        self.criteria = MSELoss()

        # Save configuration parameters.
        self.config = config
        self.prompt = self.config['prompt']
        self.seperator = self.config.get('seperator')
        self.iterations = self.config.get('iterations')
        self.ddim_steps = self.config['ddim_steps']
        self.start_guidance = self.config['start_guidance']
        self.negative_guidance = self.config['negative_guidance']
        self.image_size = self.config['image_size']
        self.lr = self.config['lr']
        self.model_config_path = self.config['model_config_path']
        self.output_dir = self.config['output_dir']

        # Retention and attack parameters.
        self.dataset_retain = self.config['dataset_retain']
        self.retain_batch = self.config['retain_batch']
        self.retain_train = self.config['retain_train']
        self.retain_step = self.config['retain_step']
        self.retain_loss_w = self.config['retain_loss_w']
        self.attack_method = self.config['attack_method']
        self.train_method = self.config['train_method']
        self.norm_layer = self.config['norm_layer']
        self.component = self.config['component']
        self.adv_prompt_num = self.config['adv_prompt_num']
        self.attack_embd_type = self.config['attack_embd_type']
        self.attack_type = self.config['attack_type']
        self.attack_init = self.config['attack_init']
        self.warmup_iter = self.config['warmup_iter']
        self.attack_step = self.config['attack_step']
        self.attack_lr = self.config['attack_lr']
        self.adv_prompt_update_step = self.config['adv_prompt_update_step']
        self.ddim_eta = self.config['ddim_eta']


        self.logger = logging.getLogger(__name__)

        attack_config = AdvAttackConfig(
            prompt="", 
            encoder_model_name_or_path=self.tokenizer.name_or_path,
            cache_path=config.get("cache_path", "./cache"),
            devices=",".join([d.strip() for d in config.get("devices", "cuda:0").split(',')]),
            attack_type=config['attack_type'],
            attack_embd_type=config['attack_embd_type'],
            attack_step=config['attack_step'],
            attack_lr=config['attack_lr'],
            attack_init=config['attack_init'],
            attack_init_embd=config['attack_init_embd'],
            attack_method=config['attack_method'],
            ddim_steps=config['ddim_steps'],
            ddim_eta=config['ddim_eta'],
            image_size=config['image_size'],
            adv_prompt_num=config['adv_prompt_num'],
            start_guidance=config['start_guidance'],
            config_path=config['model_config_path'],
            compvis_ckpt_path=None,
            backend=config['backend'],
            diffusers_model_name_or_path=config['diffusers_model_name_or_path'],
            target_ckpt=config['target_ckpt'],
            project=config.get("project_name", "default_project"),
            experiment_name=config.get("experiment_name", "default_experiment")
        )
        self.adv_attack = AdvAttack(attack_config)
        # Inject the preloaded objects
        self.adv_attack.tokenizer = self.tokenizer
        self.adv_attack.text_encoder = self.custom_text_encoder.text_encoder
        self.adv_attack.custom_text_encoder = self.custom_text_encoder
        self.adv_attack.all_embeddings = self.all_embeddings


        # Setup the dataset handler and prompt cleaning.
        self.dataset_handler = AdvUnlearnDatasetHandler(
            prompt=self.prompt,
            seperator=self.seperator,
            dataset_retain=self.dataset_retain
        )
        self.words, self.word_print = self.dataset_handler.setup_prompt()
        self.retain_dataset = self.dataset_handler.setup_dataset()

        # Initialize adversarial prompt variables.
        self.adv_word_embd = None
        self.adv_condition_embd = None
        self.adv_input_ids = None

        # Setup trainable parameters and optimizer.
        self._setup_optimizer()

    def _setup_optimizer(self):
        """
        Set up the optimizer based on the training method.
        """
        if 'text_encoder' in self.train_method:
            self.parameters = param_choices(
                model=self.custom_text_encoder,
                train_method=self.train_method,
                component=self.component,
                final_layer_norm=self.norm_layer
            )
        else:
            self.parameters = param_choices(
                model=self.model,
                train_method=self.train_method,
                component=self.component,
                final_layer_norm=self.norm_layer
            )
        self.optimizer = torch.optim.Adam(self.parameters, lr=float(self.lr))

    def encode_text(self, text):
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.devices[0])
        with torch.no_grad():
            text_embeddings = self.adv_attack.text_encoder(text_inputs.input_ids)[0]
        return text_embeddings

    def save_final_pipeline(self, output_path: str) -> None:
        """
        Save the final adversarial unlearn (adv attack) output as a full pipeline,
        containing your fine-tuned UNet, text encoder, scheduler, tokenizer, etc.
        This method packages all components (including model_index.json, safety_checker,
        feature_extractor, and scheduler configurations) into the output_path.
        """

        pipeline = StableDiffusionPipeline(
            unet=self.model,  # Final fine-tuned UNet (diffusion model)
            scheduler=self.sampler,  # Final scheduler (e.g. DDIM scheduler)
            text_encoder=self.custom_text_encoder.text_encoder,  # Fine-tuned text encoder
            tokenizer=self.tokenizer,  # Tokenizer (usually unchanged)
            vae=self.vae,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
        )
        pipeline.save_pretrained(output_path,safe_serialization=True)

    def train(self):
        ddim_eta = self.ddim_eta
        quick_sample_till_t = lambda x, s, code, batch, t: sample_model_for_diffuser(
            self.model, self.sampler,
            x, self.image_size, self.image_size, self.ddim_steps, s, ddim_eta,
            start_code=code, n_samples=batch, till_T=t, verbose=False
        )
        losses = []
        global_step = 0
        attack_round = 0 

        pbar = tqdm(range(self.iterations))
        for i in pbar:
            torch.cuda.empty_cache()
            # --- ADVERSARIAL PROMPT UPDATE ---
            if i % self.adv_prompt_update_step == 0:
                if self.retain_dataset.check_unseen_prompt_count() < self.retain_batch:
                    self.retain_dataset.reset()
                word = random.choice(self.words)
                text_input = self.tokenizer(
                    word,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                    truncation=True
                ).to(self.devices[0])
                
                # Compute text embeddings on device
                with torch.no_grad():
                    text_embeddings = id2embedding(
                        self.tokenizer,
                        self.all_embeddings,
                        text_input.input_ids,
                        self.devices[0]
                    )
                emb_0 = self.encode_text("")
                emb_p = self.encode_text(word)

                if i >= self.warmup_iter:
                    # Set models to eval for adversarial prompt update to save memory
                    self.custom_text_encoder.text_encoder.eval()
                    self.custom_text_encoder.text_encoder.requires_grad_(False)
                    self.model.eval()

                    adv_word_embd, adv_input_ids = self.adv_attack.attack(word, global_step, attack_round)
                    if self.attack_embd_type == 'word_embd':
                        self.adv_word_embd, self.adv_input_ids = adv_word_embd, adv_input_ids
                    elif self.attack_embd_type == 'condition_embd':
                        self.adv_condition_embd, self.adv_input_ids = adv_word_embd, adv_input_ids

                    global_step += self.attack_step
                    attack_round += 1 

            # --- TRAINING MODE ---
            if 'text_encoder' in self.train_method:
                self.custom_text_encoder.text_encoder.train()
                self.custom_text_encoder.text_encoder.requires_grad_(True)
                self.model.train()  # Allow gradients through conditioning
                # Optionally freeze model parameters for non-updated parts
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.custom_text_encoder.text_encoder.eval()
                self.custom_text_encoder.text_encoder.requires_grad_(False)
                self.model.train()

            self.optimizer.zero_grad()

            # --- RETENTION BRANCH ---
            if self.retain_train == 'reg':
                retain_words = self.retain_dataset.get_random_prompts(self.retain_batch)
                retain_text_inputs = self.tokenizer(
                    retain_words,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to(self.devices[0])
                with torch.no_grad():
                    retain_emb_p = self.text_encoder(retain_text_inputs.input_ids)[0]
                # Match batch size for latent generation
                retain_start_code = torch.randn((self.retain_batch, 4, self.image_size // 8, self.image_size // 8), device=self.devices[0])
                with torch.cuda.amp.autocast():
                    retain_z = quick_sample_till_t(
                        retain_emb_p,
                        self.start_guidance,
                        retain_start_code,
                        self.retain_batch,
                        int(torch.randint(self.ddim_steps, (1,), device=self.devices[0]))
                    )
            else:
                retain_emb_p = None
                retain_emb_n = None

            # --- MAIN TRAINING BRANCH ---
            # Wrap forward pass in autocast to reduce memory usage
            with torch.cuda.amp.autocast():
                if i < self.warmup_iter:
                    input_ids = text_input.input_ids
                    emb_n = self.custom_text_encoder(
                        input_ids=input_ids,
                        inputs_embeds=text_embeddings
                    )[0]
                    loss = get_train_loss_retain_diffuser(
                        self.retain_batch, self.retain_train, self.retain_loss_w,
                        self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                        emb_0, emb_p, retain_emb_p, emb_n, retain_emb_n, self.start_guidance,
                        self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                        self.image_size, self.criteria, input_ids, self.attack_embd_type
                    )
                else:
                    if self.attack_embd_type == 'word_embd':
                        loss = get_train_loss_retain_diffuser(
                            self.retain_batch, self.retain_train, self.retain_loss_w,
                            self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                            emb_0, emb_p, retain_emb_p, None, retain_emb_n, self.start_guidance,
                            self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                            self.image_size, self.criteria, self.adv_input_ids, self.attack_embd_type,
                            self.adv_word_embd
                        )
                    elif self.attack_embd_type == 'condition_embd':
                        loss = get_train_loss_retain_diffuser(
                            self.retain_batch, self.retain_train, self.retain_loss_w,
                            self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                            emb_0, emb_p, retain_emb_p, None, retain_emb_n, self.start_guidance,
                            self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                            self.image_size, self.criteria, self.adv_input_ids, self.attack_embd_type,
                            self.adv_condition_embd
                        )
            # Backward pass and optimizer step
            loss.backward()
            self.optimizer.step()
            global_step += 1

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({'Train_Loss': loss.item()}, step=global_step)
            wandb.log({'Attack_Loss': 0.0}, step=global_step)

            # Free memory for variables no longer needed
            del loss, text_input, text_embeddings, emb_n
            torch.cuda.empty_cache()

        # Save final model, etc.
        self.model.eval()
        self.custom_text_encoder.text_encoder.eval()
        self.custom_text_encoder.text_encoder.requires_grad_(False)
        if 'text_encoder' in self.train_method:
            save_text_encoder(self.output_dir, self.custom_text_encoder, self.train_method, i)
            self.logger.info(f"Output saved to {self.output_dir} dir")
        else: 
            # output_path = f"{self.output_dir}/models/diffuser_model_checkpoint_{i}"
            self.save_final_pipeline(self.output_dir)
            self.logger.info(f"Output saved to {self.output_dir} dir")
        save_history(self.output_dir, losses, self.word_print)
        return self.model


   