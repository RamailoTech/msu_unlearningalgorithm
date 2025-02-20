# mu_defense/algorithms/adv_unlearn/compvis_trainer.py

import torch
import random
import wandb
import logging

from tqdm import tqdm
from pathlib import Path
from torch.nn import MSELoss

from mu.core import BaseTrainer
from mu_defense.algorithms.adv_unlearn.utils import (
    id2embedding,
    param_choices,
    get_train_loss_retain,
    save_text_encoder,
    save_history,
    sample_model,
    save_model
)


from mu_attack.execs.adv_attack import AdvAttack
from mu_attack.configs.adv_unlearn import AdvAttackConfig
from mu_defense.algorithms.adv_unlearn import AdvUnlearnDatasetHandler


class AdvUnlearnCompvisTrainer(BaseTrainer):
    """
    Trainer for adversarial unlearning.

    This trainer performs the adversarial prompt update and retention-based
    regularized training loop for CompVis models.
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
            prompt="",  # prompt is no longer used in __init__
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
            compvis_ckpt_path=config.get("compvis_ckpt_path", ""),
            backend=config['backend'],
            diffusers_model_name_or_path="",
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

    def train(self):
        """
        Execute the adversarial unlearning training loop.
        """
        ddim_eta = self.ddim_eta
        quick_sample_till_t = lambda x, s, code, batch, t: sample_model(
            self.model, self.sampler,
            x, self.image_size, self.image_size, self.ddim_steps, s, ddim_eta,
            start_code=code, n_samples=batch, till_T=t, verbose=False
        )
        losses = []
        history = []
        global_step = 0
        attack_round = 0 

        pbar = tqdm(range(self.iterations))
        for i in pbar:
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
                )
                text_embeddings = id2embedding(
                    self.tokenizer,
                    self.all_embeddings,
                    text_input.input_ids.to(self.devices[0]),
                    self.devices[0]
                )
                # Obtain the unconditional and conditional embeddings via the original model.
                emb_0 = self.model_orig.get_learned_conditioning([''])
                emb_p = self.model_orig.get_learned_conditioning([word])

                if i >= self.warmup_iter:
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

            if 'text_encoder' in self.train_method:
                self.custom_text_encoder.text_encoder.train()
                self.custom_text_encoder.text_encoder.requires_grad_(True)
                self.model.eval()
            else:
                self.custom_text_encoder.text_encoder.eval()
                self.custom_text_encoder.text_encoder.requires_grad_(False)
                self.model.train()

            self.optimizer.zero_grad()

            if self.retain_train == 'reg':
                retain_words = self.retain_dataset.get_random_prompts(self.retain_batch)
                retain_text_input = self.tokenizer(
                    retain_words,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                    truncation=True
                )
                retain_input_ids = retain_text_input.input_ids.to(self.devices[0])
                retain_emb_p = self.model_orig.get_learned_conditioning(retain_words)
                retain_text_embeddings = id2embedding(
                    self.tokenizer,
                    self.all_embeddings,
                    retain_text_input.input_ids.to(self.devices[0]),
                    self.devices[0]
                )
                retain_text_embeddings = retain_text_embeddings.reshape(
                    self.retain_batch, -1, retain_text_embeddings.shape[-1]
                )
                retain_emb_n = self.custom_text_encoder(
                    input_ids=retain_input_ids,
                    inputs_embeds=retain_text_embeddings
                )[0]
            else:
                retain_emb_p = None
                retain_emb_n = None

            if i < self.warmup_iter:
                input_ids = text_input.input_ids.to(self.devices[0])
                emb_n = self.custom_text_encoder(
                    input_ids=input_ids,
                    inputs_embeds=text_embeddings
                )[0]
                loss = get_train_loss_retain(
                    self.retain_batch, self.retain_train, self.retain_loss_w,
                    self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                    emb_0, emb_p, retain_emb_p, emb_n, retain_emb_n, self.start_guidance,
                    self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                    self.image_size, self.criteria, input_ids, self.attack_embd_type
                )
            else:
                if self.attack_embd_type == 'word_embd':
                    loss = get_train_loss_retain(
                        self.retain_batch, self.retain_train, self.retain_loss_w,
                        self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                        emb_0, emb_p, retain_emb_p, None, retain_emb_n, self.start_guidance,
                        self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                        self.image_size, self.criteria, self.adv_input_ids, self.attack_embd_type,
                        self.adv_word_embd
                    )
                elif self.attack_embd_type == 'condition_embd':
                    loss = get_train_loss_retain(
                        self.retain_batch, self.retain_train, self.retain_loss_w,
                        self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                        emb_0, emb_p, retain_emb_p, None, retain_emb_n, self.start_guidance,
                        self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                        self.image_size, self.criteria, self.adv_input_ids, self.attack_embd_type,
                        self.adv_condition_embd
                    )
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            history.append(loss.item())
            wandb.log({'Train_Loss': loss.item()}, step=global_step)
            wandb.log({'Attack_Loss': 0.0}, step=global_step)
            global_step += 1
            self.optimizer.step()

            if self.retain_train == 'iter':
                for r in range(self.retain_step):
                    self.optimizer.zero_grad()
                    if self.retain_dataset.check_unseen_prompt_count() < self.retain_batch:
                        self.retain_dataset.reset()
                    retain_words = self.retain_dataset.get_random_prompts(self.retain_batch)
                    t_enc = torch.randint(self.ddim_steps, (1,), device=self.devices[0])
                    og_num = round((int(t_enc.item()) / self.ddim_steps) * 1000)
                    og_num_lim = round(((int(t_enc.item()) + 1) / self.ddim_steps) * 1000)
                    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.devices[0])
                    retain_start_code = torch.randn((self.retain_batch, 4, 64, 64)).to(self.devices[0])
                    retain_emb_p = self.model_orig.get_learned_conditioning(retain_words)
                    retain_z = quick_sample_till_t(
                        retain_emb_p.to(self.devices[0]),
                        self.start_guidance,
                        retain_start_code,
                        self.retain_batch,
                        int(t_enc.item())
                    )
                    retain_e_p = self.model_orig.apply_model(
                        retain_z.to(self.devices[0]),
                        t_enc_ddpm.to(self.devices[0]),
                        retain_emb_p.to(self.devices[0])
                    )
                    retain_text_input = self.tokenizer(
                        retain_words,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                        truncation=True
                    )
                    retain_input_ids = retain_text_input.input_ids.to(self.devices[0])
                    retain_text_embeddings = id2embedding(
                        self.tokenizer,
                        self.all_embeddings,
                        retain_text_input.input_ids.to(self.devices[0]),
                        self.devices[0]
                    )
                    retain_text_embeddings = retain_text_embeddings.reshape(
                        self.retain_batch, -1, retain_text_embeddings.shape[-1]
                    )
                    retain_emb_n = self.custom_text_encoder(
                        input_ids=retain_input_ids,
                        inputs_embeds=retain_text_embeddings
                    )[0]
                    retain_e_n = self.model.apply_model(
                        retain_z.to(self.devices[0]),
                        t_enc_ddpm.to(self.devices[0]),
                        retain_emb_n.to(self.devices[0])
                    )
                    retain_loss = self.criteria(
                        retain_e_n.to(self.devices[0]),
                        retain_e_p.to(self.devices[0])
                    )
                    retain_loss.backward()
                    self.optimizer.step()

            if (i + 1) % self.config['save_interval'] == 0 and (i + 1) != self.iterations and (i + 1) >= self.config['save_interval']:
                if 'text_encoder' in self.train_method:
                    save_text_encoder(self.output_dir, self.custom_text_encoder, self.train_method, i)
                else:
                    save_model(self.output_dir, self.model, self.train_method, i, save_compvis=True, save_diffusers=True, compvis_config_file=self.model_config_path, diffusers_config_file=None)
                save_history(self.output_dir, losses, self.word_print)

        self.model.eval()
        self.custom_text_encoder.text_encoder.eval()
        self.custom_text_encoder.text_encoder.requires_grad_(False)
        if 'text_encoder' in self.train_method:
            save_text_encoder(self.output_dir, self.custom_text_encoder, self.train_method, i)
            self.logger.info(f"Output saved to {self.output_dir} dir")
        else: 
            save_model(self.output_dir, self.model, self.train_method, i, save_compvis=True, save_diffusers=True, compvis_config_file=self.model_config_path, diffusers_config_file=None)
            self.logger.info(f"Output saved to {self.output_dir} dir")
        save_history(self.output_dir, losses, self.word_print)
        return self.model
