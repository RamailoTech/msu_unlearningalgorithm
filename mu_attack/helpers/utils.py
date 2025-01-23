import os
import torch
from PIL import Image
import pandas as pd
import yaml
import json
from typing import Optional, Tuple, Union

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

