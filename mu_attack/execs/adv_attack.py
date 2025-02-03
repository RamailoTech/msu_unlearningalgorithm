

import torch
from tqdm import tqdm
import random
import wandb

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

from mu_attack.configs.adv_unlearn import AdvUnlearnConfig
from mu.helpers import sample_model
from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_attack.attackers.soft_prompt import SoftPromptAttack
from mu_attack.helpers.utils import id2embedding, param_choices, get_models, retain_prompt, get_train_loss_retain,save_text_encoder, save_model, save_history



class AdvUnlearn:
    """
    Class for adversarial unlearning training.
    
    This class wraps the full training pipeline including prompt cleaning, 
    attack (adversarial prompt generation), and retention-based regularized training.
    """
    def __init__(
        self,
        config: AdvUnlearnConfig,
        **kwargs
    ):
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        
        config.validate_config()

        self.config = config
        self.prompt = config.prompt
        self.dataset_retain = config.dataset_retain
        self.retain_batch = config.retain_batch
        self.retain_train = config.retain_train
        self.retain_step = config.retain_step
        self.retain_loss_w = config.retain_loss_w
        self.attack_method = config.attack_method
        self.train_method = config.train_method
        self.norm_layer = config.norm_layer
        self.component = config.component
        self.model_name_or_path = config.model_name_or_path
        self.start_guidance = config.start_guidance
        self.negative_guidance = config.negative_guidance
        self.iterations = config.iterations
        self.save_interval = config.save_interval
        self.lr = config.lr
        self.config_path = config.config_path
        self.ckpt_path = config.ckpt_path
        self.diffusers_config_path = config.diffusers_config_path
        self.output_dir = config.output_dir
        self.devices = config.devices
        self.seperator = config.seperator
        self.image_size = config.image_size
        self.ddim_steps = config.ddim_steps
        self.adv_prompt_num = config.adv_prompt_num
        self.attack_embd_type = config.attack_embd_type
        self.attack_type = config.attack_type
        self.attack_init = config.attack_init
        self.warmup_iter = config.warmup_iter
        self.attack_step = config.attack_step
        self.attack_lr = config.attack_lr
        self.adv_prompt_update_step = config.adv_prompt_update_step
        self.ddim_eta = config.ddim_eta
        self.cache_path = config.cache_path

        # Will be set during training.
        self.words = None
        self.retain_dataset = None
        self.tokenizer = None
        self.text_encoder = None
        self.custom_text_encoder = None
        self.all_embeddings = None
        self.vae = None
        self.model_orig = None
        self.sampler_orig = None
        self.model = None
        self.sampler = None
        self.parameters = None
        self.opt = None
        self.criteria = torch.nn.MSELoss()

        # For adversarial prompt update
        self.adv_word_embd = None
        self.adv_condition_embd = None
        self.adv_input_ids = None

    def setup(self):
        """Stage 0 & 1: Prompt cleaning and training setup."""
        # --- Prompt cleaning ---
        word_print = self.prompt.replace(' ', '')
        # Special cases for certain prompts
        if self.prompt == 'allartist':
            self.prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
        if self.prompt == 'i2p':
            self.prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
        if self.prompt == "artifact":
            self.prompt = ("ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
                           "mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, "
                           "body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy")
        
        if self.seperator is not None:
            self.words = [w.strip() for w in self.prompt.split(self.seperator)]
        else:
            self.words = [self.prompt]
        print(f'The Concept Prompt to be unlearned: {self.words}')
        
        # Create a retaining dataset (assumed to be a prompt dataset)
        self.retain_dataset = retain_prompt(self.dataset_retain)
        
        # --- Training Setup ---
        ddim_eta = self.ddim_eta  # constant value for training
        
       
       
        # Load the VAE
        self.vae = AutoencoderKL.from_pretrained(self.model_name_or_path, subfolder="vae", cache_dir=self.cache_path).to(self.devices[0])
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name_or_path, subfolder="tokenizer", cache_dir=self.cache_path)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_name_or_path, subfolder="text_encoder", cache_dir=self.cache_path).to(self.devices[0])
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.devices[0])
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)
        
        # Load models using your helper function (assumed to be defined in utils)
        self.model_orig, self.sampler_orig, self.model, self.sampler = get_models(self.config_path, self.ckpt_path, self.devices)
        self.model_orig.eval()

        # Setup trainable parameters based on train_method
        if 'text_encoder' in self.train_method:
            self.parameters = param_choices(model=self.custom_text_encoder, train_method=self.train_method, component=self.component, final_layer_norm=self.norm_layer)
        else:
            self.parameters = param_choices(model=self.model, train_method=self.train_method, component=self.component, final_layer_norm=self.norm_layer)
        
        self.opt = torch.optim.Adam(self.parameters, lr=self.lr)
        
        return word_print  # For later use in saving history

    def train(self):
        """Stage 2: Training loop."""
        word_print = self.setup()
        ddim_eta = self.ddim_eta  # As used in training
        
        # A lambda function to sample until a given time step.
        quick_sample_till_t = lambda x, s, code, batch, t: sample_model(
            self.model, self.sampler,
            x, self.image_size, self.image_size, self.ddim_steps, s, ddim_eta,
            start_code=code, n_samples=batch, till_T=t, verbose=False
        )
        
        losses = []
        history = []
        global_step = 0
        attack_round = 0

        # Create a tqdm progress bar
        pbar = tqdm(range(self.iterations))
        for i in pbar:
            # --- Update adversarial prompt every adv_prompt_update_step iterations ---
            if i % self.adv_prompt_update_step == 0:
                # Reset the retaining dataset if needed
                if self.retain_dataset.check_unseen_prompt_count() < self.retain_batch:
                    self.retain_dataset.reset()
                
                # Randomly choose one prompt from the list
                word = random.sample(self.words, 1)[0]
                text_input = self.tokenizer(
                    word, padding="max_length", max_length=self.tokenizer.model_max_length,
                    return_tensors="pt", truncation=True
                )
                text_embeddings = id2embedding(self.tokenizer, self.all_embeddings, text_input.input_ids.to(self.devices[0]), self.devices[0])
                
                # Get conditional embeddings from the frozen model
                emb_0 = self.model_orig.get_learned_conditioning([''])
                emb_p = self.model_orig.get_learned_conditioning([word])
                
                # --- Attack Step: Get adversarial prompt ---
                if i >= self.warmup_iter:
                    self.custom_text_encoder.text_encoder.eval()
                    self.custom_text_encoder.text_encoder.requires_grad_(False)
                    self.model.eval()
                    
                    if attack_round == 0:
                        if self.attack_embd_type == 'word_embd':
                            self.adv_word_embd, self.adv_input_ids = SoftPromptAttack.attack(
                                global_step, word, self.model, self.model_orig, self.tokenizer,
                                self.custom_text_encoder, self.sampler, emb_0, emb_p, self.start_guidance,
                                self.devices, self.ddim_steps, ddim_eta, self.image_size, self.criteria,
                                self.adv_prompt_num, self.all_embeddings, attack_round, self.attack_type,
                                self.attack_embd_type, self.attack_step, self.attack_lr, self.attack_init,
                                None, self.attack_method
                            )
                        elif self.attack_embd_type == 'condition_embd':
                            self.adv_condition_embd, self.adv_input_ids = SoftPromptAttack.attack(
                                global_step, word, self.model, self.model_orig, self.tokenizer,
                                self.custom_text_encoder, self.sampler, emb_0, emb_p, self.start_guidance,
                                self.devices, self.ddim_steps, ddim_eta, self.image_size, self.criteria,
                                self.adv_prompt_num, self.all_embeddings, attack_round, self.attack_type,
                                self.attack_embd_type, self.attack_step, self.attack_lr, self.attack_init,
                                None, self.attack_method
                            )
                    else:
                        if self.attack_embd_type == 'word_embd':
                            self.adv_word_embd, self.adv_input_ids = SoftPromptAttack.attack(
                                global_step, word, self.model, self.model_orig, self.tokenizer,
                                self.custom_text_encoder, self.sampler, emb_0, emb_p, self.start_guidance,
                                self.devices, self.ddim_steps, ddim_eta, self.image_size, self.criteria,
                                self.adv_prompt_num, self.all_embeddings, attack_round, self.attack_type,
                                self.attack_embd_type, self.attack_step, self.attack_lr, self.attack_init,
                                self.adv_word_embd, self.attack_method
                            )
                        elif self.attack_embd_type == 'condition_embd':
                            self.adv_condition_embd, self.adv_input_ids = SoftPromptAttack.attack(
                                global_step, word, self.model, self.model_orig, self.tokenizer,
                                self.custom_text_encoder, self.sampler, emb_0, emb_p, self.start_guidance,
                                self.devices, self.ddim_steps, ddim_eta, self.image_size, self.criteria,
                                self.adv_prompt_num, self.all_embeddings, attack_round, self.attack_type,
                                self.attack_embd_type, self.attack_step, self.attack_lr, self.attack_init,
                                self.adv_condition_embd, self.attack_method
                            )
                    global_step += self.attack_step
                    attack_round += 1

            # --- Set models to training/eval modes based on training method ---
            if 'text_encoder' in self.train_method:
                self.custom_text_encoder.text_encoder.train()
                self.custom_text_encoder.text_encoder.requires_grad_(True)
                self.model.eval()
            else:
                self.custom_text_encoder.text_encoder.eval()
                self.custom_text_encoder.text_encoder.requires_grad_(False)
                self.model.train()
            self.opt.zero_grad()
            
            # --- Retaining prompts for retention regularization ---
            if self.retain_train == 'reg':
                retain_words = self.retain_dataset.get_random_prompts(self.retain_batch)
                retain_text_input = self.tokenizer(
                    retain_words, padding="max_length", max_length=self.tokenizer.model_max_length,
                    return_tensors="pt", truncation=True
                )
                retain_input_ids = retain_text_input.input_ids.to(self.devices[0])
                
                retain_emb_p = self.model_orig.get_learned_conditioning(retain_words)
                retain_text_embeddings = id2embedding(self.tokenizer, self.all_embeddings, retain_text_input.input_ids.to(self.devices[0]), self.devices[0])
                # Reshape to [batch, 77, embedding_dim]
                retain_text_embeddings = retain_text_embeddings.reshape(self.retain_batch, -1, retain_text_embeddings.shape[-1])
                retain_emb_n = self.custom_text_encoder(input_ids=retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
            else:
                retain_text_input = None
                retain_text_embeddings = None
                retain_emb_p = None
                retain_emb_n = None

            # --- Compute training loss ---
            if i < self.warmup_iter:
                # Warmup training uses the original prompt embeddings.
                input_ids = text_input.input_ids.to(self.devices[0])
                emb_n = self.custom_text_encoder(input_ids=input_ids, inputs_embeds=text_embeddings)[0]
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
                        self.image_size, self.criteria, self.adv_input_ids, self.attack_embd_type, self.adv_word_embd
                    )
                elif self.attack_embd_type == 'condition_embd':
                    loss = get_train_loss_retain(
                        self.retain_batch, self.retain_train, self.retain_loss_w,
                        self.model, self.model_orig, self.custom_text_encoder, self.sampler,
                        emb_0, emb_p, retain_emb_p, None, retain_emb_n, self.start_guidance,
                        self.negative_guidance, self.devices, self.ddim_steps, ddim_eta,
                        self.image_size, self.criteria, self.adv_input_ids, self.attack_embd_type, self.adv_condition_embd
                    )
            
            # Backpropagate loss and update weights.
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            history.append(loss.item())
            wandb.log({'Train_Loss': loss.item()}, step=global_step)
            wandb.log({'Attack_Loss': 0.0}, step=global_step)
            global_step += 1
            self.opt.step()
            
            # --- Additional Retention Training (if using iterative retention) ---
            if self.retain_train == 'iter':
                for r in range(self.retain_step):
                    print(f'==== Retain Training at step {r} ====')
                    self.opt.zero_grad()
                    if self.retain_dataset.check_unseen_prompt_count() < self.retain_batch:
                        self.retain_dataset.reset()
                    retain_words = self.retain_dataset.get_random_prompts(self.retain_batch)
                    
                    t_enc = torch.randint(self.ddim_steps, (1,), device=self.devices[0])
                    og_num = round((int(t_enc) / self.ddim_steps) * 1000)
                    og_num_lim = round((int(t_enc + 1) / self.ddim_steps) * 1000)
                    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.devices[0])
                    retain_start_code = torch.randn((self.retain_batch, 4, 64, 64)).to(self.devices[0])
                    
                    retain_emb_p = self.model_orig.get_learned_conditioning(retain_words)
                    retain_z = quick_sample_till_t(retain_emb_p.to(self.devices[0]), self.start_guidance, retain_start_code, self.retain_batch, int(t_enc))
                    retain_e_p = self.model_orig.apply_model(retain_z.to(self.devices[0]), t_enc_ddpm.to(self.devices[0]), retain_emb_p.to(self.devices[0]))
                    
                    retain_text_input = self.tokenizer(
                        retain_words, padding="max_length", max_length=self.tokenizer.model_max_length,
                        return_tensors="pt", truncation=True
                    )
                    retain_input_ids = retain_text_input.input_ids.to(self.devices[0])
                    retain_text_embeddings = id2embedding(self.tokenizer, self.all_embeddings, retain_text_input.input_ids.to(self.devices[0]), self.devices[0])
                    retain_text_embeddings = retain_text_embeddings.reshape(self.retain_batch, -1, retain_text_embeddings.shape[-1])
                    retain_emb_n = self.custom_text_encoder(input_ids=retain_input_ids, inputs_embeds=retain_text_embeddings)[0]
                    retain_e_n = self.model.apply_model(retain_z.to(self.devices[0]), t_enc_ddpm.to(self.devices[0]), retain_emb_n.to(self.devices[0]))
                    
                    retain_loss = self.criteria(retain_e_n.to(self.devices[0]), retain_e_p.to(self.devices[0]))
                    retain_loss.backward()
                    self.opt.step()
            
            # --- Checkpointing and saving history ---
            if (i + 1) % self.save_interval == 0 and (i + 1) != self.iterations and (i + 1) >= self.save_interval:
                if 'text_encoder' in self.train_method:
                    save_text_encoder(self.output_dir, self.custom_text_encoder, self.train_method, i)
                else:
                    save_model(self.output_dir, self.model, self.train_method, i, save_compvis=True,
                               save_diffusers=True, compvis_config_file=self.config_path,
                               diffusers_config_file=self.diffusers_config_path)
            if i % 1 == 0:
                save_history(self.output_dir, losses, word_print)
        
        # --- Stage 3: Save final model and loss curve ---
        self.model.eval()
        self.custom_text_encoder.text_encoder.eval()
        self.custom_text_encoder.text_encoder.requires_grad_(False)
        if 'text_encoder' in self.train_method:
            save_text_encoder(self.output_dir, self.custom_text_encoder, self.train_method, i)
        else:
            save_model(self.output_dir, self.model, self.train_method, i, save_compvis=True,
                       save_diffusers=True, compvis_config_file=self.config_path,
                       diffusers_config_file=self.diffusers_config_path)
        save_history(self.output_dir, losses, word_print)

