

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


class ClipScoreEvaluator():
    def __init__(self,
                 gen_image_path,
                #  output_path,
                 prompt_file_path,
                 devices,
                 classification_model_path="openai/clip-vit-base-patch32",
                 **kwargs):

  
        # self.output_path = output_path
        self.image_path =gen_image_path
        self.prompt_file_path = prompt_file_path
        devices = [
            d.strip() if d.strip().startswith("cuda:") else f"cuda:{int(d.strip())}"
            for d in devices[0].split(",")
        ]

        self.device = devices
        self.model_name_or_path = classification_model_path
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
        Load prompts from a file based on its extension:
        - If CSV: load from CSV file where the column name is 'prompts'
        - If LOG: load each line from the log file as a prompt
        - Else: try to load as a JSON file

        Returns:
            list: A list of prompts extracted from the file.
        """
        prompt_file_path = os.path.join(self.prompt_file_path)
        
        if not os.path.exists(prompt_file_path):
            self.logger.warning(f"No prompt file found at {prompt_file_path}. Returning an empty list.")
            return []
        
        ext = os.path.splitext(prompt_file_path)[1].lower()
        
        try:
            if ext == ".csv":
                import csv
                prompts = []
                with open(prompt_file_path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if "prompt" in row:
                            prompts.append(row["prompt"])
                self.logger.info(f"Successfully loaded {len(prompts)} prompts from CSV file {prompt_file_path}.")
                return prompts
            
            elif ext == ".log":
                prompts = []
                with open(prompt_file_path, "r", encoding="utf-8") as logfile:
                    # Assuming each line in the log file is a prompt
                    prompts = [line.strip() for line in logfile if line.strip()]
                self.logger.info(f"Successfully loaded {len(prompts)} prompts from log file {prompt_file_path}.")
                return prompts
            
            else:
                # Fall back to JSON
                with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
                    prompt_data = json.load(prompt_file)
                    prompts = [entry.get("prompt") for entry in prompt_data if "prompt" in entry]
                self.logger.info(f"Successfully loaded {len(prompts)} prompts from JSON file {prompt_file_path}.")
                return prompts
        except Exception as e:
            self.logger.error(f"An error occurred while loading prompts from {prompt_file_path}: {e}")
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

    def compute_clip_score(self, *args,**kwargs):
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
        return self.result["clip_score"]

    # def save_results(self):
    #     """
    #     Save or append the CLIP score results to a JSON file.
    #     """
    #     os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    #     existing_data = []

    #     if os.path.exists(self.output_path):
    #         try:
    #             with open(self.output_path, 'r') as json_file:
    #                 existing_data = json.load(json_file)
    #         except json.JSONDecodeError:
    #             pass  # Ignore if the file is invalid

    #     if isinstance(existing_data, list):
    #         existing_data.append(self.result)
    #     elif isinstance(existing_data, dict):
    #         existing_data.update(self.result)
    #     else:
    #         existing_data = [self.result]

    #     with open(self.output_path, 'w') as json_file:
    #         json.dump(existing_data, json_file, indent=4)

    #     self.logger.info(f'Results saved to {self.output_path}')


    def run(self,*args, **kwargs):
        """
        Run the CLIP score evaluator.
        """
        self.logger.info("Calculating Clip score...")

        # Load and prepare data
        self.load_and_prepare_data()

        # Compute CLIP score
        self.compute_clip_score()

        # Save results
        # self.save_results()


