
import os
import timm
import torch
import sys
import glob

import json

import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from evaluation.helpers.parser import get_parser

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available

def preprocess_image(device, image: Image.Image):
    """
    Preprocess the input PIL image before feeding into the classifier.
    Replicates the transforms from your accuracy.py script.
    """
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return image_transform(image).unsqueeze(0).to(device)

def load_categories(reference_dir: str) -> list:
    """
    Load unique categories from a CSV file in the 'prompts' folder under the given reference directory.
    
    Args:
        reference_dir (str): The base directory where the 'prompts' folder is located.
        
    Returns:
        List[str]: A sorted list of unique categories extracted from the CSV file.
        
    Raises:
        FileNotFoundError: If no CSV file is found in the prompts folder or if the file doesn't exist.
    """
    prompts_folder = os.path.join(reference_dir, 'prompts')
    prompts_files = glob.glob(os.path.join(prompts_folder, '*.csv'))
    if not prompts_files:
        raise FileNotFoundError(f"No CSV file found in the prompts folder: {prompts_folder}")
    
    prompts_file = prompts_files[0]
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    data = pd.read_csv(prompts_file)
    unique_categories = set()
    for cats in data['categories']:
        if isinstance(cats, str):
            for cat in cats.split(','):
                unique_categories.add(cat.strip())
    categories = sorted(list(unique_categories))
    return categories

def convert_time(time_str):
    time_parts = time_str.split(":")
    hours, minutes, seconds_microseconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
    total_minutes_direct = hours * 60 + minutes + seconds_microseconds / 60
    return total_minutes_direct


def load_experiments(root):
    """Helper method to load experiments."""
    exps = []
    for e in os.listdir(root):
        try:
            exps.append(get_parser(root))
        except Exception as ex:
            print(f'Failed to parse {e}: {ex}')
    return exps

def load_prompts(prompt_file_path):
    """
    Load prompts from a file based on its extension:
    - If CSV: load from CSV file where the column name is 'prompts'
    - If LOG: load each line from the log file as a prompt
    - Else: try to load as a JSON file

    Returns:
        list: A list of prompts extracted from the file.
    """
    prompt_file_path = os.path.join(prompt_file_path)
    
    if not os.path.exists(prompt_file_path):
        print(f"No prompt file found at {prompt_file_path}. Returning an empty list.")
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
            print(f"Successfully loaded {len(prompts)} prompts from CSV file {prompt_file_path}.")
            return prompts
        elif ext == ".json":
            # Fall back to JSON
            with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
                prompt_data = json.load(prompt_file)
                prompts = [entry.get("prompt") for entry in prompt_data if "prompt" in entry]
            print(f"Successfully loaded {len(prompts)} prompts from JSON file {prompt_file_path}.")
            return prompts
    except Exception as e:
        print(f"An error occurred while loading prompts from {prompt_file_path}: {e}")
        return []

def load_and_prepare_images(image_path,target_size=(224, 224)):
        """
        Convert all images in a folder to NumPy arrays.
        image_path: path to images
        Args:
            folder_path (str): Path to the folder containing images.
            target_size (tuple): Desired image size (height, width) for resizing. Default is (224, 224).

        Returns:
            list: A list of NumPy arrays representing the images.
        """
        image_arrays = []

        # Loop through each file in the folder
        for filename in os.listdir(image_path):
            file_path = os.path.join(image_path, filename)

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
                print(f"Error loading image {filename}: {e}")

        return np.array(image_arrays)

def load_model(classifier_ckpt_path, device ,classification_model = "vit_large_patch16_224",task="class"):
    """
    Load the classification model for evaluation, using 'timm' 
    or any approach you prefer. 
    We assume your config has 'ckpt_path' and 'task' keys, etc.
    """
    print("Loading classification model...")
    model = timm.create_model(
        classification_model, 
        pretrained=True
    ).to(device)
    task = task # "style" or "class"
    num_classes = len(theme_available) if task == "style" else len(class_available)
    model.head = torch.nn.Linear(1024, num_classes).to(device)

    # Load checkpoint
    ckpt_path = classifier_ckpt_path
    print(f"Loading classification checkpoint from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.eval()
    print("Classification model loaded successfully.")
    return model