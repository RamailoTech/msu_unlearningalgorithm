# mu_attack/attackers/soft_prompt.py

import torch
import wandb

from mu_attack.helpers.utils import split_id, id2embedding, split_embd, init_adv, construct_embd, construct_id, sample_model, sample_model_for_diffuser


class SoftPromptAttack:
    """
    A class to perform a soft prompt attack on the ESD model.
    
    Attributes:
        model: The ESD model.
        model_orig: The frozen (original) model.
        tokenizer: The tokenizer.
        text_encoder: The text encoder.
        sampler: The sampler (or scheduler) used for diffusion.
        emb_0: Unconditional embedding.
        emb_p: Conditional embedding.
        start_guidance: Guidance scale for sampling.
        devices: List of devices to use.
        ddim_steps: Number of DDIM steps.
        ddim_eta: The eta parameter for DDIM.
        image_size: The size (width and height) for generated images.
        criteria: The loss criteria function.
        k: Number of tokens (or a related parameter for the prompt).
        all_embeddings: The preloaded word embeddings.
        backend: String indicating which backend is used ("compvis" or "diffusers").
    """
    
    def __init__(self, model, model_orig, tokenizer, text_encoder, sampler,
                 emb_0, emb_p, start_guidance, devices, ddim_steps, ddim_eta,
                 image_size, criteria, k, all_embeddings, backend="compvis"):
        self.model = model
        self.model_orig = model_orig
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.sampler = sampler
        self.emb_0 = emb_0
        self.emb_p = emb_p
        self.start_guidance = start_guidance
        self.devices = devices
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.image_size = image_size
        self.criteria = criteria
        self.k = k
        self.all_embeddings = all_embeddings
        self.backend = backend

    def attack(self, global_step, word, attack_round, attack_type,
               attack_embd_type, attack_step, attack_lr,
               attack_init=None, attack_init_embd=None, attack_method='pgd'):
        """
        Perform soft prompt attack on the ESD model.

        Args:
            global_step (int): The current global training step.
            word (str): The input prompt.
            attack_round (int): The current attack round.
            attack_type (str): Type of attack ("add" or "insert").
            attack_embd_type (str): Type of adversarial embedding ("condition_embd" or "word_embd").
            attack_step (int): Number of steps to run the attack.
            attack_lr (float): Learning rate for the adversarial optimization.
            attack_init (str, optional): Initialization method ("latest" or "random").
            attack_init_embd (torch.Tensor, optional): Initial adversarial embedding.
            attack_method (str, optional): Attack method to use ("pgd" or "fast_at").
            
        Returns:
            tuple: Depending on attack_embd_type, returns a tuple (embedding, input_ids)
                   where the embedding is either a conditional or word embedding.
        """
        orig_prompt_len = len(word.split())
        if attack_type == 'add':
            # When using "add", update k to match the prompt length.
            self.k = orig_prompt_len

        # A helper lambda to sample an image until a given time step.
        if self.backend == "compvis":
            quick_sample_till_t = lambda x, s, code, t: sample_model(
                self.model, self.sampler, x, self.image_size, self.image_size,
                self.ddim_steps, s, self.ddim_eta, start_code=code, till_T=t, verbose=False
            )
        elif self.backend == "diffusers":
            quick_sample_till_t = lambda x, s, code, t: sample_model_for_diffuser(
            self.model, self.sampler, x, self.image_size, self.image_size,
            self.ddim_steps, s, self.ddim_eta, start_code=code, till_T=t, verbose=False
        )
        
        # --- Tokenization and Embedding ---
        text_input = self.tokenizer(
            word, padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt", truncation=True
        )
        sot_id, mid_id, replace_id, eot_id = split_id(
            text_input.input_ids.to(self.devices[0]), self.k, orig_prompt_len
        )
        
        text_embeddings = id2embedding(
            self.tokenizer, self.all_embeddings,
            text_input.input_ids.to(self.devices[0]), self.devices[0]
        )
        sot_embd, mid_embd, _, eot_embd = split_embd(text_embeddings, self.k, orig_prompt_len)
        
        # --- Initialize the adversarial embedding ---
        if attack_init == 'latest':
            adv_embedding = init_adv(self.k, self.tokenizer, self.all_embeddings,
                                     attack_type, self.devices[0], 1, attack_init_embd)
        elif attack_init == 'random':
            adv_embedding = init_adv(self.k, self.tokenizer, self.all_embeddings,
                                     attack_type, self.devices[0], 1)
        else:
            # Default initialization if no method is provided
            adv_embedding = init_adv(self.k, self.tokenizer, self.all_embeddings,
                                     attack_type, self.devices[0], 1)
        
        attack_opt = torch.optim.Adam([adv_embedding], lr=attack_lr)
        
        # For the condition_embd attack type, construct the initial adversarial condition embedding.
        if attack_embd_type == 'condition_embd':
            input_adv_condition_embedding = construct_embd(
                self.k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd
            )
            adv_input_ids = construct_id(
                self.k, replace_id, attack_type, sot_id, eot_id, mid_id
            )
        
        print(f'[{attack_type}] Starting {attack_method} attack on "{word}"')
        
        # --- Attack Loop ---
        for i in range(attack_step):
            # Randomly sample a time step for the attack.
            t_enc = torch.randint(self.ddim_steps, (1,), device=self.devices[0])
            og_num = round((int(t_enc) / self.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / self.ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.devices[0])
            start_code = torch.randn((1, 4, 64, 64)).to(self.devices[0])
            
            with torch.no_grad():
                # Sample a latent z using the conditional embedding.
                z = quick_sample_till_t(
                    self.emb_p.to(self.devices[0]), self.start_guidance, start_code, int(t_enc)
                )
                if self.backend == "compvis":
                    # For compvis, use apply_model to get the noise predictions.
                    e_0 = self.model_orig.apply_model(
                        z.to(self.devices[0]), t_enc_ddpm.to(self.devices[0]), self.emb_0.to(self.devices[0])
                    )
                    e_p = self.model_orig.apply_model(
                        z.to(self.devices[0]), t_enc_ddpm.to(self.devices[0]), self.emb_p.to(self.devices[0])
                    )
                elif self.backend == "diffusers":
                    # For diffusers, call the UNet directly with encoder_hidden_states.
                    out_0 = self.model_orig(
                        z.to(self.devices[0]),
                        t_enc_ddpm.to(self.devices[0]),
                        encoder_hidden_states=self.emb_0.to(self.devices[0])
                    )
                    e_0 = out_0.sample if hasattr(out_0, "sample") else out_0
                    out_p = self.model_orig(
                        z.to(self.devices[0]),
                        t_enc_ddpm.to(self.devices[0]),
                        encoder_hidden_states=self.emb_p.to(self.devices[0])
                    )
                    e_p = out_p.sample if hasattr(out_p, "sample") else out_p
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
            
            # For word_embd attack type, update the adversarial condition embedding using the text encoder.
            if attack_embd_type == 'word_embd':
                input_adv_word_embedding = construct_embd(
                    self.k, adv_embedding, attack_type, sot_embd, mid_embd, eot_embd
                )
                adv_input_ids = construct_id(
                    self.k, replace_id, attack_type, sot_id, eot_id, mid_id
                )
                input_adv_condition_embedding = self.text_encoder(
                    input_ids=adv_input_ids.to(self.devices[0]),
                    inputs_embeds=input_adv_word_embedding
                )[0]
            
            # Get the conditional score from the model with the adversarial condition embedding.
            if self.backend == "compvis":
                e_n = self.model.apply_model(
                    z.to(self.devices[0]), t_enc_ddpm.to(self.devices[0]),
                    input_adv_condition_embedding.to(self.devices[0])
                )
            elif self.backend == "diffusers":
                out_n = self.model(
                    z.to(self.devices[0]),
                    t_enc_ddpm.to(self.devices[0]),
                    encoder_hidden_states=input_adv_condition_embedding.to(self.devices[0])
                )
                e_n = out_n.sample if hasattr(out_n, "sample") else out_n
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
            # Prevent gradients on the frozen branch.
            e_0.requires_grad = False
            e_p.requires_grad = False
            
            # Compute the loss between the adversarial output and the target.
            loss = self.criteria(e_n.to(self.devices[0]), e_p.to(self.devices[0]))
            loss.backward()
            
            if attack_method == 'pgd':
                attack_opt.step()
            elif attack_method == 'fast_at':
                adv_embedding.grad.sign_()
                attack_opt.step()
            else:
                raise ValueError('attack_method must be either pgd or fast_at')
            
            wandb.log({'Attack_Loss': loss.item()}, step=global_step + i)
            wandb.log({'Train_Loss': 0.0}, step=global_step + i)
            print(f'Step: {global_step + i}, Attack_Loss: {loss.item()}')
            print(f'Step: {global_step + i}, Train_Loss: 0.0')
        
        # --- Return the adversarial embeddings and input IDs ---
        if attack_embd_type == 'condition_embd':
            return input_adv_condition_embedding, adv_input_ids
        elif attack_embd_type == 'word_embd':
            return input_adv_word_embedding, adv_input_ids
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')
