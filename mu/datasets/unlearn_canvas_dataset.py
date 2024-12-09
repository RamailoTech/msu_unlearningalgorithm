# datasets/unlearn_canvas_dataset.py

from typing import Any, Tuple
from PIL import Image
import os
import torch
import numpy as np
from einops import rearrange
from datasets.base_dataset import BaseDataset
from torchvision import transforms
from datasets.constants import (
    uc_theme_available,
    uc_class_available,
    uc_sample_theme_available,
    uc_sample_class_available
)

class UnlearnCanvasDataset(BaseDataset):
    """
    Dataset for UnlearnCanvas algorithm.
    Allows selection of specific themes and classes.
    """
    def __init__(
        self,
        data_dir: str,
        selected_theme: str,
        selected_class: str,
        use_sample: bool = False,
        transform: Any = None
    ):
        """
        Initialize the UnlearnCanvasDataset.

        Args:
            data_dir (str): Root directory containing dataset.
            selected_theme (str): Theme to filter images.
            selected_class (str): Class to filter images.
            use_sample (bool, optional): Whether to use sample constants. Defaults to False.
            transform (Any, optional): Transformations to apply to the images.
        """
        super().__init__()
        
        if use_sample:
            assert selected_theme in uc_sample_theme_available, (
                f"Selected theme '{selected_theme}' is not available in sample themes."
            )
            assert selected_class in uc_sample_class_available, (
                f"Selected class '{selected_class}' is not available in sample classes."
            )
        else:
            assert selected_theme in uc_theme_available, (
                f"Selected theme '{selected_theme}' is not available."
            )
            assert selected_class in uc_class_available, (
                f"Selected class '{selected_class}' is not available."
            )

        self.selected_theme = selected_theme
        self.selected_class = selected_class
        self.transform = transform

        # Paths to images and prompts
        # self.images_txt = os.path.join(data_dir, selected_theme, selected_class, 'images.txt')
        # self.prompts_txt = os.path.join(data_dir, selected_theme, selected_class, 'prompts.txt')
        self.images_txt = os.path.join(data_dir, 'images.txt')
        self.prompts_txt = os.path.join(data_dir, 'prompts.txt')

        # Check if files exist
        if not os.path.exists(self.images_txt):
            raise FileNotFoundError(f"images.txt not found at {self.images_txt}")
        if not os.path.exists(self.prompts_txt):
            raise FileNotFoundError(f"prompts.txt not found at {self.prompts_txt}")

        # Load image paths and prompts
        self.image_paths = self.read_text_lines(self.images_txt)
        self.prompts = self.read_text_lines(self.prompts_txt)

        assert len(self.image_paths) == len(self.prompts), (
            "Number of images and prompts must be equal."
        )

    def read_text_lines(self, path: str):
        """
        Read lines from a text file.

        Args:
            path (str): Path to the text file.

        Returns:
            List[str]: List of lines.
        """
        with open(path, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, str]: A tuple containing the data sample and its corresponding prompt.
        """
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = self.prompts[idx]

        if self.transform:
            image = self.transform(image)

        # Convert the image to tensor and normalize
        image = rearrange(
            2 * torch.tensor(np.array(image)).float() / 255 - 1,
            "h w c -> c h w"
        )

        return image, prompt
