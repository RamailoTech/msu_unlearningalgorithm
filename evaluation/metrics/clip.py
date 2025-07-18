

# evaluation/metrics/clip.py

from typing import Any, Dict
import logging
import numpy as np
import os
import json
from functools import partial
from PIL import Image

import torch
from torchmetrics.functional.multimodal import clip_score as calculate_clip_score

from evaluation.helpers.utils import load_prompts, load_and_prepare_images




def clip_score(images="data/coco_reference_images/coco_10k", prompt_path="data/prompts/coco_10k.csv", device="cuda", model_name_or_path="openai/clip-vit-base-patch32"):
    """
    Calculate the CLIP score for the given images and prompts using a specified model.

    Parameters:
        images (numpy.ndarray): The input images in NumPy array format. By default it uses the COCO dataset.
        prompt_path (path to prompt):json or csv file. By default it uses the COCO dataset.
        device (list or tuple): The device(s) to use (e.g., ['cuda:0'] or ['cpu']).
        model_name_or_path (str): The model name or path for the CLIP model.

    Returns:
        float: The computed CLIP score rounded to four decimal places.
    """
    clip_score_fn = partial(calculate_clip_score, model_name_or_path=model_name_or_path)
    if "0" in device and torch.cuda.is_available():
        device = [
            f"cuda:{int(d.strip())}" for d in device.split(",")
        ]
    else:
        device = ["cpu"]
    images = load_and_prepare_images(images)
    image_tensor = torch.from_numpy(images).to(device[0])
    prompts = load_prompts(prompt_path)
    clip_score_value = clip_score_fn(image_tensor, prompts).detach()
    
    return round(float(clip_score_value), 4)

