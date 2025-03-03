# mu_attack/helpers/utils.py

import sys
import os
import pandas as pd
import random
import yaml
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)


current_dir = Path(__file__).parent.parent.parent

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

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
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/imagenet243_retain.csv')
    elif dataset_retain == 'imagenet243_no_filter':
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/imagenet243_no_filter_retain.csv')
    elif dataset_retain == 'coco_object':
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/coco_object_retain.csv')
    elif dataset_retain == 'coco_object_no_filter':
        retain_dataset = PromptDataset(current_dir / 'data/prompts/train/coco_object_no_filter_retain.csv')
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

