import os
from typing import Any, Dict, List
from torch.utils.data import DataLoader
from algorithms.scissorhands.datasets.scissorhands_dataset import ScissorHandsDataset

from algorithms.scissorhands.logger import setup_logger
from core.base_data_handler import BaseDataHandler

from datasets.constants import * 

class ScissorHandsDataHandler(BaseDataHandler):
    """
    Concrete data handler for the ScissorHands algorithm.
    Manages forget and remain datasets through ScissorHandsDataset.
    """
    
    def __init__(
        self,
        original_data_dir: str,
        new_data_dir: str,
        selected_theme: str,
        selected_class: str,
        batch_size: int = 4,
        image_size: int = 512,
        interpolation: str = 'bicubic',
        use_sample: bool = False,
    ):
        """
        Initialize the ScissorHandsDataHandler.

        Args:
            original_data_dir (str): Directory containing the original dataset organized by themes and classes.
            new_data_dir (str): Directory where the new datasets will be saved.
            selected_theme (str): Theme to filter images.
            selected_class (str): Class to filter images.
            batch_size (int, optional): Batch size for data loaders. Defaults to 4.
            image_size (int, optional): Size to resize images. Defaults to 512.
            interpolation (str, optional): Interpolation mode. Defaults to 'bicubic'.
            use_sample (bool, optional): Whether to use sample datasets. Defaults to False.
            num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory in DataLoader. Defaults to True.
        """
        self.original_data_dir = original_data_dir
        self.new_data_dir = new_data_dir
        self.selected_theme = selected_theme
        self.selected_class = selected_class
        self.batch_size = batch_size
        self.image_size = image_size
        self.interpolation = interpolation
        self.use_sample = use_sample

        # Initialize logger
        self.logger = setup_logger('ScissorHandsDataHandler')

        # Generate the dataset upon initialization
        self.generate_dataset()

        # Initialize DataLoaders
        self.data_loaders = self.get_data_loaders()

    def generate_dataset(self):
        """
        Generate datasets by organizing images into themes and classes.
        This method encapsulates the dataset generation logic.
        """
        self.logger.info("Starting dataset generation...")
        # For style unlearning
        for theme in uc_sample_theme_available:
            theme_dir = os.path.join(self.new_data_dir, theme)
            os.makedirs(theme_dir, exist_ok=True)
            prompt_list = []
            path_list = []
            for class_ in uc_sample_class_available:
                for idx in [1, 2, 3]:
                    prompt = f"A {class_} image in {theme.replace('_', ' ')} style."
                    image_path = os.path.join(self.original_data_dir, theme, class_, f"{idx}.jpg")
                    if os.path.exists(image_path):
                        prompt_list.append(prompt)
                        path_list.append(image_path)
                    else:
                        self.logger.warning(f"Image not found: {image_path}")
            # Write prompts and images to text files
            prompts_txt_path = os.path.join(theme_dir, 'prompts.txt')
            images_txt_path = os.path.join(theme_dir, 'images.txt')
            with open(prompts_txt_path, 'w') as f:
                f.write('\n'.join(prompt_list))
            with open(images_txt_path, 'w') as f:
                f.write('\n'.join(path_list))
            self.logger.info(f"Generated dataset for theme '{theme}' with {len(path_list)} samples.")

        # For Seed Images
        seed_theme = "Seed_Images"
        seed_dir = os.path.join(self.new_data_dir, seed_theme)
        os.makedirs(seed_dir, exist_ok=True)
        prompt_list = []
        path_list = []
        for class_ in uc_sample_class_available:
            for idx in [1, 2, 3]:
                prompt = f"A {class_} image in Photo style."
                image_path = os.path.join(self.original_data_dir, seed_theme, class_, f"{idx}.jpg")
                if os.path.exists(image_path):
                    prompt_list.append(prompt)
                    path_list.append(image_path)
                else:
                    self.logger.warning(f"Image not found: {image_path}")
        # Write prompts and images to text files
        prompts_txt_path = os.path.join(seed_dir, 'prompts.txt')
        images_txt_path = os.path.join(seed_dir, 'images.txt')
        with open(prompts_txt_path, 'w') as f:
            f.write('\n'.join(prompt_list))
        with open(images_txt_path, 'w') as f:
            f.write('\n'.join(path_list))
        self.logger.info(f"Generated Seed Images dataset with {len(path_list)} samples.")

        # For class unlearning
        for object_class in uc_sample_class_available:
            class_dir = os.path.join(self.new_data_dir, object_class)
            os.makedirs(class_dir, exist_ok=True)
            prompt_list = []
            path_list = []
            for theme in uc_sample_theme_available:
                for idx in [1, 2, 3]:
                    prompt = f"A {object_class} image in {theme.replace('_', ' ')} style."
                    image_path = os.path.join(self.original_data_dir, theme, object_class, f"{idx}.jpg")
                    if os.path.exists(image_path):
                        prompt_list.append(prompt)
                        path_list.append(image_path)
                    else:
                        self.logger.warning(f"Image not found: {image_path}")
            # Write prompts and images to text files
            prompts_txt_path = os.path.join(class_dir, 'prompts.txt')
            images_txt_path = os.path.join(class_dir, 'images.txt')
            with open(prompts_txt_path, 'w') as f:
                f.write('\n'.join(prompt_list))
            with open(images_txt_path, 'w') as f:
                f.write('\n'.join(path_list))
            self.logger.info(f"Generated dataset for class '{object_class}' with {len(path_list)} samples.")

        self.logger.info("Dataset generation completed.")

    def load_data(self, data_path: str) -> Any:
        """
        Load data from the specified path.
        For ScissorHands, this involves loading image paths and prompts.

        Args:
            data_path (str): Path to the data.

        Returns:
            Any: Loaded data (e.g., dictionary containing image paths and prompts).
        """
        images_txt = os.path.join(data_path, 'images.txt')
        prompts_txt = os.path.join(data_path, 'prompts.txt')
        if not os.path.isfile(images_txt) or not os.path.isfile(prompts_txt):
            self.logger.error(f"Missing images.txt or prompts.txt in {data_path}")
            raise FileNotFoundError(f"Missing images.txt or prompts.txt in {data_path}")
        image_paths = self.read_text_lines(images_txt)
        prompts = self.read_text_lines(prompts_txt)
        if len(image_paths) != len(prompts):
            self.logger.error(f"Mismatch between images and prompts in {data_path}")
            raise ValueError(f"Mismatch between images and prompts in {data_path}")
        return {'image_paths': image_paths, 'prompts': prompts}

    def preprocess_data(self, data: Any) -> Any:
        """
        Preprocess the data.
        For ScissorHands, this is handled by the Dataset class via transformations.

        Args:
            data (Any): Raw data to preprocess.

        Returns:
            Any: Preprocessed data.
        """
        # No additional preprocessing required as transformations are applied within the Dataset
        return data

    def get_data_loaders(self, batch_size: int = None) -> Dict[str, DataLoader]:
        """
        Get data loaders for forget and remain datasets.

        Args:
            batch_size (int, optional): Batch size for data loaders. If None, uses self.batch_size.

        Returns:
            Dict[str, DataLoader]: Dictionary containing 'forget' and 'remain' data loaders.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Determine dataset directories based on whether sample is used
        if self.use_sample:
            forget_data_dir = os.path.join(self.new_data_dir, self.selected_theme)
            remain_data_dir = os.path.join(self.new_data_dir, "Seed_Images")
        else:
            forget_data_dir = os.path.join(self.new_data_dir, self.selected_theme)
            remain_data_dir = os.path.join(self.new_data_dir, "Seed_Images")
        # Initialize ScissorHandsDataset
        scissorhands_dataset = ScissorHandsDataset(
            forget_data_dir=forget_data_dir,
            remain_data_dir=remain_data_dir,
            selected_theme=self.selected_theme,
            selected_class=self.selected_class,
            use_sample=self.use_sample,
            image_size=self.image_size,
            interpolation=self.interpolation
        )

        # Retrieve DataLoaders
        data_loaders = scissorhands_dataset.get_data_loaders()

        return data_loaders

    @staticmethod
    def read_text_lines(path: str) -> List[str]:
        """Read lines from a text file and strip whitespace."""
        with open(path, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines