
# mu_defense/algorithms/adv_unlearn/utils.py

import sys
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from mu.helpers import load_model_from_config

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler

current_dir = Path(__file__).parent.parent.parent.parent

class PromptDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.unseen_indices = list(self.data.index)  # 保存所有未见过的索引

    def get_random_prompts(self, num_prompts=1):
        # Ensure that the number of prompts requested is not greater than the number of unseen prompts
        num_prompts = min(num_prompts, len(self.unseen_indices))

        # Randomly select num_prompts indices from the list of unseen indices
        selected_indices = random.sample(self.unseen_indices, num_prompts)
        
        # Remove the selected indices from the list of unseen indices
        for index in selected_indices:
            self.unseen_indices.remove(index)

        # return the prompts corresponding to the selected indices
        return self.data.loc[selected_indices, 'prompt'].tolist()

    def has_unseen_prompts(self):
        # check if there are any unseen prompts
        return len(self.unseen_indices) > 0
    
    def reset(self):
        self.unseen_indices = list(self.data.index)
        
    def check_unseen_prompt_count(self):
        return len(self.unseen_indices)

def retain_prompt(dataset_retain):
    # Prompt Dataset to be retained

    if dataset_retain == 'imagenet243':
        retain_dataset = PromptDataset(current_dir /'data/prompts/train/imagenet243_retain.csv')
    elif dataset_retain == 'imagenet243_no_filter':
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/imagenet243_no_filter_retain.csv')
    elif dataset_retain == 'coco_object':
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/coco_object_retain.csv')
    elif dataset_retain == 'coco_object_no_filter':
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/coco_object_no_filter_retain.csv')
    else:
        raise ValueError('Invalid dataset for retaining prompts')
    
    return retain_dataset

def id2embedding(tokenizer, all_embeddings, input_ids, device):
    input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(tokenizer.get_vocab())).float()
    input_one_hot = torch.unsqueeze(input_one_hot,0).to(device)
    input_embeds = input_one_hot @ all_embeddings
    return input_embeds


def get_models_for_diffusers(diffuser_model_name_or_path, devices, target_ckpt=None, cache_path=None):
    """
    Loads two copies of a Diffusers UNet model along with their DDIM schedulers,
    and additionally loads the VAE, safety checker (disabled by default), and feature extractor.
    
    Args:
        diffuser_model_name_or_path (str): The Hugging Face model identifier or local path.
        target_ckpt (str or None): Path to a target checkpoint to load into the primary model (on devices[0]).
                                   If None, no state dict is loaded.
        devices (list or tuple): A list/tuple of devices, e.g. [device0, device1].
        cache_path (str or None): Optional cache directory for pretrained weights.
        
    Returns:
        model_orig: The UNet loaded on devices[1].
        sampler_orig: The DDIM scheduler corresponding to model_orig.
        model: The UNet loaded on devices[0] (optionally updated with target_ckpt).
        sampler: The DDIM scheduler corresponding to model.
        vae: The AutoencoderKL (VAE) loaded on devices[0].
        safety_checker: Disabled (None) by default.
        feature_extractor: The feature extractor from the pipeline.
    """
    
    # Load the original UNet model (used for e.g. computing loss) on devices[1]
    model_orig = UNet2DConditionModel.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="unet",
        cache_dir=cache_path
    ).to(devices[1])
    
    # Create a DDIM scheduler for the original model
    sampler_orig = DDIMScheduler.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="scheduler",
        cache_dir=cache_path
    )
    
    # Load the second copy of the UNet model on devices[0]
    model = UNet2DConditionModel.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="unet",
        cache_dir=cache_path
    ).to(devices[0])
    
    # Optionally load a target checkpoint into the model
    if target_ckpt is not None:
        state_dict = torch.load(target_ckpt, map_location=devices[0])
        model.load_state_dict(state_dict)
    
    # Create a DDIM scheduler for the second model
    sampler = DDIMScheduler.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="scheduler",
        cache_dir=cache_path
    )
    
    # Load the VAE model on devices[0]
    vae = AutoencoderKL.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="vae",
        cache_dir=cache_path
    ).to(devices[0])
    
    # Load safety_checker and feature_extractor via a temporary pipeline.
    # We disable the safety_checker (to avoid _init_weights errors) by setting safety_checker=None.
    pipe = StableDiffusionPipeline.from_pretrained(
        diffuser_model_name_or_path,
        cache_dir=cache_path,
        safety_checker=None  # Disable safety_checker; you can implement your own dummy if needed.
    )
    safety_checker = None  # Alternatively, set this to a dummy safety checker.
    feature_extractor = pipe.feature_extractor
    
    return model_orig, sampler_orig, model, sampler, vae, safety_checker, feature_extractor


def get_models_for_compvis(config_path, compvis_ckpt_path, devices):
    model_orig = load_model_from_config(config_path, compvis_ckpt_path, devices[0])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, compvis_ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    # return model_orig, sampler_orig
    return model_orig, sampler_orig, model, sampler



@torch.no_grad()
def sample_model_for_diffuser(model, scheduler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None,
                              n_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
    """
    Diffusers-compatible sampling function.
    """
    device = c.device

    # If no starting code is provided, sample random noise.
    if start_code is None:
        start_code = torch.randn((n_samples, 4, h // 8, w // 8), device=device)
    latents = start_code

    # Set the number of timesteps in the scheduler.
    scheduler.set_timesteps(ddim_steps)

    # Prepare conditioning.
    if scale != 1.0:
        # Create unconditional embeddings (all zeros for this example).
        uc = torch.zeros_like(c)
        # Duplicate only the conditioning inputs.
        c_in = torch.cat([uc, c], dim=0)
    else:
        c_in = c

    # Diffusion sampling loop.
    for t in scheduler.timesteps:
        # Scale the latents as required by the scheduler.
        latent_model_input = scheduler.scale_model_input(latents, t)
        if scale != 1.0:
            # Duplicate the latent model input for classifier-free guidance.
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
        # Run the model.
        model_output = model(latent_model_input, t, encoder_hidden_states=c_in)
        # Process the output.
        if scale != 1.0:
            noise_pred_uncond, noise_pred_text = model_output.sample.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = model_output.sample

        # Update the latents with the original (unduplicated) latents.
        latents = scheduler.step(noise_pred, t, latents).prev_sample


    return latents





@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


def get_train_loss_retain( retain_batch, retain_train, retain_loss_w, model, model_orig, text_encoder, sampler, emb_0, emb_p, retain_emb_p,  emb_n, retain_emb_n, start_guidance, negative_guidance, devices, ddim_steps, ddim_eta, image_size, criteria, adv_input_ids, attack_embd_type, adv_embd=None):
    """_summary_

    Args:
        model: ESD model
        model_orig: frozen DDPM model
        sampler: DDIMSampler for DDPM model
        
        emb_0: unconditional embedding
        emb_p: conditional embedding (for ground truth concept)
        emb_n: conditional embedding (for modified concept)
        
        start_guidance: unconditional guidance for ESD model
        negative_guidance: negative guidance for ESD model
        
        devices: list of devices for ESD and DDPM models 
        ddim_steps: number of steps for DDIMSampler
        ddim_eta: eta for DDIMSampler
        image_size: image size for DDIMSampler
        
        criteria: loss function for ESD model
        
        adv_input_ids: input_ids for adversarial word embedding
        adv_emb_n: adversarial conditional embedding
        adv_word_emb_n: adversarial word embedding

    Returns:
        loss: training loss for ESD model
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, n_samples=batch, till_T=t, verbose=False)
    
    
    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
    if retain_train == 'reg':
        retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])
    
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, 1, int(t_enc)) # emb_p seems to work better instead of emb_0
        # get conditional and unconditional scores from frozen model at time step t and image z
        e_0 = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]))
        e_p = model_orig.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_p.to(devices[0]))
        
        if retain_train == 'reg':
            retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch, int(t_enc)) # emb_p seems to work better instead of emb_0
            # retain_e_0 = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_0.to(devices[0]))
            retain_e_p = model_orig.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_p.to(devices[0]))
    
    if adv_embd is None:
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))
    else:
        if attack_embd_type == 'condition_embd':
            # Train with adversarial conditional embedding
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), adv_embd.to(devices[0]))
        elif attack_embd_type == 'word_embd':
            # Train with adversarial word embedding
            print('====== Training with adversarial word embedding =====')
            adv_emb_n = text_encoder(input_ids = adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), adv_emb_n.to(devices[0]))
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')
    
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
    # loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
    
    # return loss

    if retain_train == 'reg':
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        print('====== Training with retain batch =====')
        unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
        
        retain_e_n = model.apply_model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), retain_emb_n.to(devices[0]))
        
        # retain_e_0.requires_grad = False
        retain_e_p.requires_grad = False
        retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
        
        loss = unlearn_loss + retain_loss_w * retain_loss
        return loss
        
    else:
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) 
        return unlearn_loss


def get_train_loss_retain_diffuser(
    retain_batch, retain_train, retain_loss_w,
    model, model_orig, text_encoder, sampler,
    emb_0, emb_p, retain_emb_p, emb_n, retain_emb_n,
    start_guidance, negative_guidance, devices,
    ddim_steps, ddim_eta, image_size, criteria,
    adv_input_ids, attack_embd_type, adv_embd=None
):
    """
    Compute the training loss for unlearning (with retaining) using Diffusers.
    
    Args:
        Same as get_train_loss_retain.
    
    Returns:
        loss: The computed training loss.
    """
    quick_sample_till_t = lambda x, s, code, batch, t: sample_model_for_diffuser(
        model, sampler, x, image_size, image_size, ddim_steps, s, ddim_eta,
        start_code=code, n_samples=batch, till_T=t, verbose=False
    )
    
    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # Time step mapping
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round(((int(t_enc) + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
    
    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
    if retain_train == 'reg':
        retain_start_code = torch.randn((retain_batch, 4, 64, 64)).to(devices[0])
    
    with torch.no_grad():
        # Sample latent z using the Diffuser backend
        z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, 1, int(t_enc))
        
        # Get outputs from the frozen model for conditional/unconditional embeddings
        out_0 = model_orig(z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=emb_0.to(devices[0]))
        e_0 = out_0.sample if hasattr(out_0, "sample") else out_0
        
        out_p = model_orig(z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=emb_p.to(devices[0]))
        e_p = out_p.sample if hasattr(out_p, "sample") else out_p
        
        if retain_train == 'reg':
            retain_z = quick_sample_till_t(retain_emb_p.to(devices[0]), start_guidance, retain_start_code, retain_batch, int(t_enc))
            out_retain = model_orig(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=retain_emb_p.to(devices[0]))
            retain_e_p = out_retain.sample if hasattr(out_retain, "sample") else out_retain
    
    if adv_embd is None:
        out_n = model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=emb_n.to(devices[0]))
        e_n = out_n.sample if hasattr(out_n, "sample") else out_n
    else:
        if attack_embd_type == 'condition_embd':
            out_n = model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=adv_embd.to(devices[0]))
            e_n = out_n.sample if hasattr(out_n, "sample") else out_n
        elif attack_embd_type == 'word_embd':
            print('====== Training with adversarial word embedding =====')
            adv_emb_n = text_encoder(input_ids=adv_input_ids.to(devices[0]), inputs_embeds=adv_embd.to(devices[0]))[0]
            out_n = model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=adv_emb_n.to(devices[0]))
            e_n = out_n.sample if hasattr(out_n, "sample") else out_n
        else:
            raise ValueError('attack_embd_type must be either condition_embd or word_embd')
    
    # Ensure gradients are not computed for frozen outputs
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    if retain_train == 'reg':
        # Reconstruction loss for unlearning and retention
        print('====== Training with retain batch =====')
        unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance * (e_p.to(devices[0]) - e_0.to(devices[0]))))
        
        out_retain_n = model(retain_z.to(devices[0]), t_enc_ddpm.to(devices[0]), encoder_hidden_states=retain_emb_n.to(devices[0]))
        retain_e_n = out_retain_n.sample if hasattr(out_retain_n, "sample") else out_retain_n
        
        retain_e_p.requires_grad = False
        retain_loss = criteria(retain_e_n.to(devices[0]), retain_e_p.to(devices[0]))
        
        loss = unlearn_loss + retain_loss_w * retain_loss
        return loss
    else:
        # Reconstruction loss for unlearning only
        unlearn_loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance * (e_p.to(devices[0]) - e_0.to(devices[0]))))
        return unlearn_loss


def param_choices(model, train_method, component='all', final_layer_norm=False):
    # choose parameters to train based on train_method
    parameters = []
    
    # Text Encoder FUll Weight Tuning
    if train_method == 'text_encoder_full':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Final Layer Norm
            if name.startswith('final_layer_norm'):
                if component == 'all' or final_layer_norm==True:
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            # Transformer layers 
            elif name.startswith('encoder'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            # Embedding layers
            else:
                pass
           
    # Text Encoder Layer 0 Tuning
    elif train_method == 'text_encoder_layer0':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer012':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer01234':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer012345':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123456':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01234567':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012345678':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer0123456789':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012345678910':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer01234567891011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.3') or name.startswith('encoder.layers.4') or name.startswith('encoder.layers.5') or name.startswith('encoder.layers.6') or name.startswith('encoder.layers.7') or name.startswith('encoder.layers.8') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    elif train_method == 'text_encoder_layer0_11':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
    
    
    elif train_method == 'text_encoder_layer01_1011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass
            
    elif train_method == 'text_encoder_layer012_91011':
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith('encoder.layers.0') or name.startswith('encoder.layers.1') or name.startswith('encoder.layers.2') or name.startswith('encoder.layers.9') or name.startswith('encoder.layers.10') or name.startswith('encoder.layers.11'):
                if component == 'ffn' and 'mlp' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'attn' and 'self_attn' in name:
                    print(name)
                    parameters.append(param)
                elif component == 'all':
                    print(name)
                    parameters.append(param)
                else:
                    pass
            
            elif name.startswith('final_layer_norm') and final_layer_norm==True:
                print(name)
                parameters.append(param)
            
            else:
                pass

    # UNet Model Tuning
    else:
        # For CompVis-style models, the diffusion model is nested in model.model.diffusion_model.
        # For Diffusers models (e.g. UNet2DConditionModel), the model is the UNet.
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            unet = model.model.diffusion_model
        else:
            unet = model

        for name, param in unet.named_parameters():
            # Train all layers except x-attentions and time_embed layers
            if train_method == 'noxattn':
                if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                    continue
                else:
                    print(name)
                    parameters.append(param)
                    
            # Train only self-attention layers
            elif train_method == 'selfattn':
                if 'attn1' in name:
                    print(name)
                    parameters.append(param)
                    
            # Train only x-attention layers
            elif train_method == 'xattn':
                if 'attn2' in name:
                    print(name)
                    parameters.append(param)
                    
            # Train all layers
            elif train_method == 'full':
                print(name)
                parameters.append(param)
                
            # Train all layers except time embed layers
            elif train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    print(name)
                    parameters.append(param)
                    
            # Train specific x-layer blocks
            elif train_method == 'xlayer':
                if 'attn2' in name and ('output_blocks.6.' in name or 'output_blocks.8.' in name):
                    print(name)
                    parameters.append(param)
                    
            # Train specific self-layer blocks
            elif train_method == 'selflayer':
                if 'attn1' in name and ('input_blocks.4.' in name or 'input_blocks.7.' in name):
                    print(name)
                    parameters.append(param)
    
    return parameters

def save_text_encoder(folder_path, model, name, num):
    # SAVE MODEL

    # PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'
    folder_path = f'{folder_path}/models'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/TextEncoder-{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/TextEncoder-{name}.pt'
    
    torch.save(model.state_dict(), path)



def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params
    vae_params = original_config.model.params.first_stage_config.params.ddconfig

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

    head_dim = unet_params.num_heads if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params.use_linear_in_transformer if "use_linear_in_transformer" in unet_params else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim = [5, 10, 20, 20]

    config = dict(
        sample_size=image_size // vae_scale_factor,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=head_dim,
        use_linear_projection=use_linear_projection,
    )

    return config

def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping

def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])

def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())

    unet_key = "model.diffusion_model."
    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100 and extract_ema:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        print(
            "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
            " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
        )
        for key in keys:
            if key.startswith("model.diffusion_model"):
                flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
    else:
        if sum(k.startswith("model_ema") for k in keys) > 100:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )

        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

def save_history(folder_path, losses, word_print):
    folder_path = f'{folder_path}/logs'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)


def save_model(folder_path, model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

    # PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'
    folder_path = f'{folder_path}/models'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/Compvis-UNet-{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/Compvis-UNet-{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format') 
        savemodelDiffusers(path, name, compvis_config_file, diffusers_config_file, device=device )

def savemodelDiffusers(path, name, compvis_config_file, diffusers_config_file, device='cpu'):
    checkpoint_path = path

    original_config_file = compvis_config_file
    config_file = diffusers_config_file
    num_in_channels = 4
    scheduler_type = 'ddim'
    pipeline_type = None
    image_size = 512
    prediction_type = 'epsilon'
    extract_ema = False
    dump_path = path.replace('Compvis','Diffusers')
    upcast_attention = False


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Sometimes models don't have the global_step item
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]
    else:
        print("global_step key not found in model")
        global_step = None

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    upcast_attention = upcast_attention
    if original_config_file is None:
        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"

        if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
            if not os.path.isfile("v2-inference-v.yaml"):
                # model_type = "v2"
                os.system(
                    "wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
                    " -O v2-inference-v.yaml"
                )
            original_config_file = "./v2-inference-v.yaml"

            if global_step == 110000:
                # v2.1 needs to upcast attention
                upcast_attention = True
        else:
            if not os.path.isfile("v1-inference.yaml"):
                # model_type = "v1"
                os.system(
                    "wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
                    " -O v1-inference.yaml"
                )
            original_config_file = "./v1-inference.yaml"

    original_config = OmegaConf.load(original_config_file)

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        if image_size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            image_size = 512 if global_step == 875000 else 768
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 512

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end
    scheduler = DDIMScheduler(
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        beta_start=beta_start,
        num_train_timesteps=num_train_timesteps,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    if scheduler_type == "pndm":
        config = dict(scheduler.config)
        config["skip_prk_steps"] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == "ddim":
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = False
    unet = UNet2DConditionModel(**unet_config)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
    )
    torch.save(converted_unet_checkpoint, dump_path)    

