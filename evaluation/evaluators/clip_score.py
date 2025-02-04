# mu_attack/evaluators/clip_score.py

from typing import Any, Dict
import logging
import numpy as np
import os
import json
from functools import partial
from PIL import Image

import torch
from torchmetrics.functional.multimodal import clip_score
import torch.nn.functional as F

from mu.core.base_config import BaseConfig
from mu_attack.configs.evaluation import AttackEvaluatorConfig
from evaluation.core import AttackBaseEvaluator


class ClipScoreEvaluator(AttackBaseEvaluator):
    def __init__(self, config: AttackEvaluatorConfig, **kwargs):
        super().__init__(config, **kwargs)

        for key, value in kwargs.items():
            if not hasattr(config, key):
                setattr(config, key, value)
                continue
            config_attr = getattr(config, key)
            if isinstance(config_attr, BaseConfig) and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    setattr(config_attr, sub_key, sub_val)
            elif isinstance(config_attr, dict) and isinstance(value, dict):
                config_attr.update(value)
            else:
                setattr(config, key, value)
        
        self.config = config.to_dict()
        config.validate_config()
        self.output_path = self.config.get('output_path')
        self.config = self.config.get("clip", {})
        self.image_path = self.config['image_path']
        self.log_path = self.config['log_path']
        devices = [
            f"cuda:{int(d.strip())}" for d in self.config.get("devices", "0").split(",")
        ]
        self.device = devices
        self.model_name_or_path = self.config['model_name_or_path']

        # Pass the correct model name or path
        self.clip_score_fn = partial(clip_score, model_name_or_path=self.model_name_or_path)
        self.result = {}
        self.logger = logging.getLogger(__name__)

        self.prompt = self.load_prompts()

    
    def calculate_clip_score(self, images, prompts, device):
        clip_score = self.clip_score_fn(torch.from_numpy(images).to(device[0]), prompts).detach()
        return round(float(clip_score), 4)

    def load_prompts(self):
        """
        Load prompts from a JSON file and return them as a list.

        Returns:
            list: A list of prompts extracted from the JSON file.
        """
        prompt_file_path = os.path.join(self.log_path)

        if not os.path.exists(prompt_file_path):
            self.logger.warning(f"No prompt JSON file found at {prompt_file_path}. Returning an empty list.")
            return []

        try:
            with open(prompt_file_path, "r") as prompt_file:
                prompt_data = json.load(prompt_file)

                # Extract the 'prompt' field from each entry in the JSON
                prompts = [entry.get("prompt") for entry in prompt_data if "prompt" in entry]
                self.logger.info(f"Successfully loaded {len(prompts)} prompts from {prompt_file_path}.")
                return prompts
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON file {prompt_file_path}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"An error occurred while loading prompts: {e}")
            return []

    
    def load_and_prepare_data(self,target_size=(224, 224)):
        """
        Convert all images in a folder to NumPy arrays.
        
        Args:
            folder_path (str): Path to the folder containing images.
            target_size (tuple): Desired image size (height, width) for resizing. Default is (224, 224).

        Returns:
            list: A list of NumPy arrays representing the images.
        """
        image_arrays = []

        # Loop through each file in the folder
        for filename in os.listdir(self.image_path):
            file_path = os.path.join(self.image_path, filename)

            try:
                # Open the image and resize it
                with Image.open(file_path) as img:
                    img = img.convert("RGB")  # Ensure all images are RGB
                    img_resized = img.resize(target_size)
                    
                    # Convert to NumPy array and normalize between 0-1
                    img_array = np.array(img_resized).astype(np.float32) / 255.0
                    
                    # Move channel to the front (C, H, W) if needed for models
                    img_array = np.transpose(img_array, (2, 0, 1))  # Shape: (3, height, width)
                    
                    image_arrays.append(img_array)
            except Exception as e:
                self.logger.error(f"Error loading image {filename}: {e}")

        return np.array(image_arrays)

    def compute_score(self, *args,**kwargs):
        """
        Calculate the CLIP score for an image and a prompt.
        
        Args:
            image (PIL.Image): The image for evaluation.
            prompt (str): The text prompt for comparison.

        Returns:
            float: The calculated CLIP score.
        """
        image = self.load_and_prepare_data()

        # Calculate CLIP score using torchmetrics
        self.result["clip_score"] = self.calculate_clip_score(image, self.prompt,self.device)
        self.logger.info(f"CLIP score: {self.result['clip_score']}")

    def save_results(self):
        """
        Save or append the CLIP score results to a JSON file.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        existing_data = []

        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            except json.JSONDecodeError:
                pass  # Ignore if the file is invalid

        if isinstance(existing_data, list):
            existing_data.append(self.result)
        elif isinstance(existing_data, dict):
            existing_data.update(self.result)
        else:
            existing_data = [self.result]

        with open(self.output_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        self.logger.info(f'Results saved to {self.output_path}')


    def run(self,*args, **kwargs):
        """
        Run the CLIP score evaluator.
        """
        self.logger.info("Calculating Clip score...")

        # Load and prepare data
        self.load_and_prepare_data()

        # Compute CLIP score
        self.compute_score()

        # Save results
        self.save_results()
