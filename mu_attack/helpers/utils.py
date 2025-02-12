import os
import pandas as pd
import random
import yaml

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms

from diffusers import UNet2DConditionModel, DDIMScheduler


from mu.helpers.utils import load_model_from_config
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler



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
        retain_dataset = PromptDataset('data/prompts/train/imagenet243_retain.csv')
    elif dataset_retain == 'imagenet243_no_filter':
        retain_dataset = PromptDataset('data/prompts/train/imagenet243_no_filter_retain.csv')
    elif dataset_retain == 'coco_object':
        retain_dataset = PromptDataset('data/prompts/train/coco_object_retain.csv')
    elif dataset_retain == 'coco_object_no_filter':
        retain_dataset = PromptDataset('data/prompts/train/coco_object_no_filter_retain.csv')
    else:
        raise ValueError('Invalid dataset for retaining prompts')
    
    return retain_dataset

def load_config(yaml_path):
    """Loads the configuration from a YAML file."""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    return {}


def _convert_image_to_rgb(image):
    '''
    Convert image to RGB if it is grayscale
    '''
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

class PNGImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        prompts_df = pd.read_csv(os.path.join(self.root_dir,'prompts.csv'))
        try:
            self.data = prompts_df[['prompt', 'evaluation_seed', 'evaluation_guidance']] if 'evaluation_seed' in prompts_df.columns else prompts_df[['prompt']]
        except:
            self.data = prompts_df[['prompt', 'evaluation_seed']] if 'evaluation_seed' in prompts_df.columns else prompts_df[['prompt']]
        self.idxs = [i for i in range(len(self.data))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        # image = TF.to_tensor(image)
        prompt = self.data.iloc[idx].prompt
        seed = self.data.iloc[idx].evaluation_seed if 'evaluation_seed' in self.data.columns else None
        guidance_scale = self.data.iloc[idx].evaluation_guidance if 'evaluation_guidance' in self.data.columns else 7.5  
        return None, prompt, seed, guidance_scale

def get_dataset(root_dir):
    return PNGImageDataset(root_dir=root_dir,transform=get_transform()) 

def convert_time(time_str):
    time_parts = time_str.split(":")
    hours, minutes, seconds_microseconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
    total_minutes_direct = hours * 60 + minutes + seconds_microseconds / 60
    return total_minutes_direct

def id2embedding(tokenizer, all_embeddings, input_ids, device):
    input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(tokenizer.get_vocab())).float()
    input_one_hot = torch.unsqueeze(input_one_hot,0).to(device)
    input_embeds = input_one_hot @ all_embeddings
    return input_embeds

def split_id(input_ids, k, orig_prompt_len):
    sot_id, mid_id, replace_id, eot_id = torch.split(input_ids, [1, orig_prompt_len, k, 76-orig_prompt_len-k], dim=1)
    return sot_id, mid_id, replace_id, eot_id

def split_embd(input_embed, k, orig_prompt_len):
    sot_embd, mid_embd, replace_embd, eot_embd = torch.split(input_embed, [1, orig_prompt_len, k, 76-orig_prompt_len-k ], dim=1)
    return sot_embd, mid_embd, replace_embd, eot_embd

def init_adv(k, tokenizer, all_embeddings, attack_type, device, batch = 1, attack_init_embd = None):
    # Different attack types have different initializations (Attack types: add, insert)
    adv_embedding = torch.nn.Parameter(torch.randn([batch, k, 768])).to(device)
    
    if attack_init_embd is not None:
        # Use the provided initial adversarial embedding
        adv_embedding.data = attack_init_embd[:,1:1+k].data
    else:
        # Random sample k words from the vocabulary as the initial adversarial words
        tmp_ids = torch.randint(0,len(tokenizer),(batch, k)).to(device)
        tmp_embeddings = id2embedding(tokenizer, all_embeddings, tmp_ids, device)
        tmp_embeddings = tmp_embeddings.reshape(batch, k, 768)
        adv_embedding.data = tmp_embeddings.data
    adv_embedding = adv_embedding.detach().requires_grad_(True)
    
    return adv_embedding

def construct_embd(k, adv_embedding, insertion_location, sot_embd, mid_embd, eot_embd):
    if insertion_location == 'prefix_k':     # Prepend k words before the original prompt
        embedding = torch.cat([sot_embd,adv_embedding,mid_embd,eot_embd],dim=1)
    elif insertion_location == 'replace_k':  # Replace k words in the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,mid_embd.shape[1],1)
        embedding = torch.cat([sot_embd,adv_embedding,replace_embd,eot_embd],dim=1)
    elif insertion_location == 'add':      # Add perturbation to the original prompt
        replace_embd = eot_embd[:,0,:].repeat(1,k,1)
        embedding = torch.cat([sot_embd,adv_embedding+mid_embd,replace_embd,eot_embd],dim=1)
    elif insertion_location == 'suffix_k':   # Append k words after the original prompt
        embedding = torch.cat([sot_embd,mid_embd,adv_embedding,eot_embd],dim=1)
    elif insertion_location == 'mid_k':      # Insert k words in the middle of the original prompt
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        embedding.append(mid_embd[:,:total_num//2,:])
        embedding.append(adv_embedding)
        embedding.append(mid_embd[:,total_num//2:,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
    elif insertion_location == 'insert_k':   # seperate k words into the original prompt with equal intervals
        embedding = [sot_embd,]
        total_num = mid_embd.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            embedding.append(mid_embd[:,internals*i:internals*(i+1),:])
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
        embedding.append(mid_embd[:,internals*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
        
    elif insertion_location == 'per_k_words':
        embedding = [sot_embd,]
        for i in range(adv_embedding.size(1) - 1):
            embedding.append(adv_embedding[:,i,:].unsqueeze(1))
            embedding.append(mid_embd[:,3*i:3*(i+1),:])
        embedding.append(adv_embedding[:,-1,:].unsqueeze(1))
        embedding.append(mid_embd[:,3*(i+1):,:])
        embedding.append(eot_embd)
        embedding = torch.cat(embedding,dim=1)
    return embedding

def construct_id(k, adv_id, insertion_location,sot_id,eot_id,mid_id):
    if insertion_location == 'prefix_k':
        input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        
    elif insertion_location == 'replace_k':
        replace_id = eot_id[:,0].repeat(1,mid_id.shape[1])
        input_ids = torch.cat([sot_id,adv_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'add':
        replace_id = eot_id[:,0].repeat(1,k)
        input_ids = torch.cat([sot_id,mid_id,replace_id,eot_id],dim=1)
    
    elif insertion_location == 'suffix_k':
        input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
        
    elif insertion_location == 'mid_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        input_ids.append(mid_id[:,:total_num//2])
        input_ids.append(adv_id)
        input_ids.append(mid_id[:,total_num//2:])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'insert_k':
        input_ids = [sot_id,]
        total_num = mid_id.size(1)
        internals = total_num // (k+1)
        for i in range(k):
            input_ids.append(mid_id[:,internals*i:internals*(i+1)])
            input_ids.append(adv_id[:,i].unsqueeze(1))
        input_ids.append(mid_id[:,internals*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
        
    elif insertion_location == 'per_k_words':
        input_ids = [sot_id,]
        for i in range(adv_id.size(1) - 1):
            input_ids.append(adv_id[:,i].unsqueeze(1))
            input_ids.append(mid_id[:,3*i:3*(i+1)])
        input_ids.append(adv_id[:,-1].unsqueeze(1))
        input_ids.append(mid_id[:,3*(i+1):])
        input_ids.append(eot_id)
        input_ids = torch.cat(input_ids,dim=1)
    return input_ids



def get_models_for_compvis(config_path, compvis_ckpt_path, devices):
    model_orig = load_model_from_config(config_path, compvis_ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, compvis_ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def get_models_for_diffusers(diffuser_model_name_or_path, target_ckpt, devices, cache_path=None):
    """
    Loads two copies of a Diffusers UNet model along with their DDIM schedulers.
    
    Args:
        model_name_or_path (str): The Hugging Face model identifier or local path.
        target_ckpt (str or None): Path to a target checkpoint to load into the primary model (on devices[0]).
                                   If None, no state dict is loaded.
        devices (list or tuple): A list/tuple of two devices, e.g. [device0, device1].
        cache_path (str or None): Optional cache directory for pretrained weights.
        
    Returns:
        model_orig: The UNet loaded on devices[1].
        sampler_orig: The DDIM scheduler corresponding to model_orig.
        model: The UNet loaded on devices[0] (optionally updated with target_ckpt).
        sampler: The DDIM scheduler corresponding to model.
    """
    
    # Load the original model (used for e.g. computing loss, etc.) on devices[1]
    model_orig = UNet2DConditionModel.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="unet",
        cache_dir=cache_path
    ).to(devices[1])
    
    # Create a DDIM scheduler for model_orig. (Note: diffusers DDIMScheduler is used here;
    # adjust the subfolder or configuration if your scheduler is stored elsewhere.)
    sampler_orig = DDIMScheduler.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="scheduler",
        cache_dir=cache_path
    )
    
    # Load the second copy of the model on devices[0]
    model = UNet2DConditionModel.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="unet",
        cache_dir=cache_path
    ).to(devices[0])
    
    # Optionally load a target checkpoint into model
    if target_ckpt is not None:
        state_dict = torch.load(target_ckpt, map_location=devices[0])
        model.load_state_dict(state_dict)
    
    sampler = DDIMScheduler.from_pretrained(
        diffuser_model_name_or_path,
        subfolder="scheduler",
        cache_dir=cache_path
    )
    
    return model_orig, sampler_orig, model, sampler

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

@torch.no_grad()
def sample_model_for_diffuser(model, scheduler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None,
                 n_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
    """
    Diffusers-compatible sampling function.

    Args:
        model: The UNet model (from diffusers).
        scheduler: A DDIMScheduler (or similar) instance.
        c (torch.Tensor): The conditional encoder_hidden_states.
        h (int): Image height.
        w (int): Image width.
        ddim_steps (int): Number of diffusion steps.
        scale (float): Guidance scale. If not 1.0, classifier-free guidance is applied.
        ddim_eta (float): The eta parameter for DDIM (unused in this basic implementation).
        start_code (torch.Tensor, optional): Starting latent code. If None, random noise is used.
        n_samples (int): Number of samples to generate.
        t_start, log_every_t, till_T, verbose: Additional parameters (not used in this diffusers implementation).

    Returns:
        torch.Tensor: The generated latent sample.
    """
    device = c.device

    # If no starting code is provided, sample random noise.
    if start_code is None:
        start_code = torch.randn((n_samples, 4, h // 8, w // 8), device=device)
    latents = start_code

    # Set the number of timesteps in the scheduler.
    scheduler.set_timesteps(ddim_steps)

    # If using classifier-free guidance, prepare unconditional embeddings.
    if scale != 1.0:
        # In a full implementation you would obtain these from your text encoder
        # For this example, we simply create a tensor of zeros with the same shape as c.
        uc = torch.zeros_like(c)
        # Duplicate latents and conditioning for guidance.
        latents = torch.cat([latents, latents], dim=0)
        c_in = torch.cat([uc, c], dim=0)
    else:
        c_in = c

    # Diffusion sampling loop.
    for t in scheduler.timesteps:
        # Scale the latents as required by the scheduler.
        latent_model_input = scheduler.scale_model_input(latents, t)
        model_output = model(latent_model_input, t, encoder_hidden_states=c_in)
        # Assume model_output is a ModelOutput with a 'sample' attribute.
        if scale != 1.0:
            # Split the batch into unconditional and conditional parts.
            noise_pred_uncond, noise_pred_text = model_output.sample.chunk(2)
            # Apply classifier-free guidance.
            noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = model_output.sample

        # Step the scheduler.
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # If guidance was used, return only the second half of the batch.
    if scale != 1.0:
        latents = latents[n_samples:]
    return latents