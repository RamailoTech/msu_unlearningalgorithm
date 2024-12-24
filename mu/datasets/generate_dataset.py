import os
from typing import List
from pathlib import Path
from PIL import Image
import random
import shutil

from mu.helpers.logger import setup_logger


def generate_dataset(
    original_data_dir: str,
    new_dir: str,
    theme_available: List[str],
    class_available: List[str],
    seed_images_theme: str,
    num_images_per_class: int = 3,
):
    """
    Generate datasets by organizing images into themes and classes.

    Args:
        original_data_dir (str): Directory containing the original dataset organized by themes and classes.
        new_dir (str): Directory where the new datasets will be saved.
        theme_available (List[str]): List of themes to include.
        class_available (List[str]): List of classes to include.
        seed_images_theme (str): Theme name for seed images.
        num_images_per_class (int, optional): Number of images per class. Defaults to 3.
    """
    logger = setup_logger("GenerateDataset")
    os.makedirs(new_dir, exist_ok=True)
    logger.info(f"Generating datasets in {new_dir}")

    for theme in theme_available:
        theme_dir = os.path.join(new_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)
        logger.info(f"Processing theme: {theme}")

        for class_name in class_available:
            class_dir = os.path.join(theme_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            logger.info(f"  Processing class: {class_name}")

            # Path to original class directory
            original_class_dir = os.path.join(original_data_dir, theme, class_name)
            if not os.path.isdir(original_class_dir):
                logger.warning(
                    f"Original class directory not found: {original_class_dir}"
                )
                continue

            # List all image files in the original class directory
            image_files = [
                f
                for f in os.listdir(original_class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]

            if not image_files:
                logger.warning(f"No images found in {original_class_dir}")
                continue

            # Randomly select images if necessary
            selected_images = random.sample(
                image_files, min(num_images_per_class, len(image_files))
            )

            for img_file in selected_images:
                src_path = os.path.join(original_class_dir, img_file)
                dst_path = os.path.join(class_dir, img_file)
                shutil.copy(src_path, dst_path)
                logger.info(f"    Copied {src_path} to {dst_path}")

    # Generate seed images if necessary
    if seed_images_theme in theme_available:
        seed_dir = os.path.join(new_dir, seed_images_theme)
        os.makedirs(seed_dir, exist_ok=True)
        logger.info(f"Processing seed images for theme: {seed_images_theme}")

        # Assuming seed images are under a specific class, e.g., 'Seed_Images'
        seed_class_dir = os.path.join(
            original_data_dir, seed_images_theme, "Seed_Images"
        )
        if not os.path.isdir(seed_class_dir):
            logger.warning(
                f"Original seed images directory not found: {seed_class_dir}"
            )
        else:
            seed_images = [
                f
                for f in os.listdir(seed_class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]

            if not seed_images:
                logger.warning(f"No seed images found in {seed_class_dir}")
            else:
                selected_seed_images = random.sample(
                    seed_images, min(num_images_per_class, len(seed_images))
                )

                for img_file in selected_seed_images:
                    src_path = os.path.join(seed_class_dir, img_file)
                    dst_path = os.path.join(seed_dir, img_file)
                    shutil.copy(src_path, dst_path)
                    logger.info(f"  Copied seed image {src_path} to {dst_path}")
