import os
from typing import List
from pathlib import Path
from PIL import Image
import random

from mu.datasets.generic_dataset import GenericImageDataset
from mu.datasets.utils import get_logger

def generate_dataset(
    original_data_dir: str,
    new_dir: str,
    theme_available: List[str],
    class_available: List[str],
    seed_images_theme: str,
    num_images_per_class: int = 3
):
    """
    Generate datasets by organizing images and prompts into text files.

    Args:
        original_data_dir (str): Directory containing the original dataset.
        new_dir (str): Directory where the new datasets will be saved.
        theme_available (List[str]): List of available themes.
        class_available (List[str]): List of available classes.
        seed_images_theme (str): Theme name for seed images.
        num_images_per_class (int, optional): Number of images per class. Defaults to 3.
    """
    logger = get_logger('GenerateDataset')
    os.makedirs(new_dir, exist_ok=True)
    logger.info(f"Generating datasets in {new_dir}")

    for theme in theme_available:
        theme_dir = os.path.join(new_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)
        images_txt = os.path.join(theme_dir, 'images.txt')
        prompts_txt = os.path.join(theme_dir, 'prompts.txt')

        # Assuming original_data_dir contains subdirectories for each theme
        original_theme_dir = os.path.join(original_data_dir, theme)
        if not os.path.isdir(original_theme_dir):
            logger.warning(f"Original theme directory not found: {original_theme_dir}")
            continue

        # List all image files in the original theme directory
        image_files = [os.path.join(original_theme_dir, f) for f in os.listdir(original_theme_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # Randomly select a subset if necessary
        selected_images = image_files[:num_images_per_class]

        # Write image paths to images.txt
        with open(images_txt, 'w') as img_f, open(prompts_txt, 'w') as prompt_f:
            for img_path in selected_images:
                img_f.write(f"{img_path}\n")
                # Generate prompts based on theme and class
                # This is a placeholder. Replace with actual prompt generation logic.
                prompt = f"A {theme.lower()} image."
                prompt_f.write(f"{prompt}\n")

        logger.info(f"Generated {len(selected_images)} samples for theme '{theme}'")

    # Generate seed images if necessary
    seed_dir = os.path.join(new_dir, seed_images_theme)
    os.makedirs(seed_dir, exist_ok=True)
    images_txt = os.path.join(seed_dir, 'images.txt')
    prompts_txt = os.path.join(seed_dir, 'prompts.txt')

    # Assuming seed_images_theme directory exists in original_data_dir
    original_seed_dir = os.path.join(original_data_dir, seed_images_theme)
    if not os.path.isdir(original_seed_dir):
        logger.warning(f"Original seed theme directory not found: {original_seed_dir}")
    else:
        image_files = [os.path.join(original_seed_dir, f) for f in os.listdir(original_seed_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        selected_images = image_files[:num_images_per_class]

        with open(images_txt, 'w') as img_f, open(prompts_txt, 'w') as prompt_f:
            for img_path in selected_images:
                img_f.write(f"{img_path}\n")
                prompt = "A seed image."
                prompt_f.write(f"{prompt}\n")

        logger.info(f"Generated {len(selected_images)} seed samples.")
