import os
import torch
from PIL import Image
import pandas as pd
import yaml
import json
from typing import Optional, Tuple, Union

from mu.helpers.utils import load_model_from_config
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torchvision.transforms as torch_transforms



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
        for name, param in model.model.diffusion_model.named_parameters():
            # train all layers except x-attns and time_embed layers
            if train_method == 'noxattn':
                if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                    pass
                else:
                    print(name)
                    parameters.append(param)
                    
            # train only self attention layers
            if train_method == 'selfattn':
                if 'attn1' in name:
                    print(name)
                    parameters.append(param)
                    
            # train only x attention layers
            if train_method == 'xattn':
                if 'attn2' in name:
                    print(name)
                    parameters.append(param)
                    
            # train all layers
            if train_method == 'full':
                print(name)
                parameters.append(param)
                
            # train all layers except time embed layers
            if train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    print(name)
                    parameters.append(param)
            if train_method == 'xlayer':
                if 'attn2' in name:
                    if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                        print(name)
                        parameters.append(param)
            if train_method == 'selflayer':
                if 'attn1' in name:
                    if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                        print(name)
                        parameters.append(param)
    
    return parameters


def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler