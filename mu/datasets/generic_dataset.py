# mu/datasets/generic_dataset.py

from typing import Any, Tuple, List
from PIL import Image
import os
import torch
import numpy as np
from einops import rearrange

from mu.datasets import BaseDataset
from mu.helpers import read_text_lines

class GenericDataset(BaseDataset):
    """
    Generic Dataset.
    Allows selection of specific categories based on provided options.
    """
    def __init__(
        self,
        data_dir: str,
        template_name: str,
        categories: List[str],
        transform: Any = None
    ):
        """
        Initialize the GenericDataset.

        Args:
            data_dir (str): Root directory containing the dataset.
            template_name (str): Name of the template to use (e.g., a specific category).
            categories (List[str]): List of available category options.
            transform (Any, optional): Transformations to apply to the images.
        """
        super().__init__()

        # Validate that the chosen template is in the list of available categories
        assert template_name in categories, (
            f"Selected template name '{template_name}' is not available. "
            f"Available options are: {categories}"
        )

        self.template_name = template_name
        self.transform = transform

        # Paths to the images and prompts text files
        self.images_txt = os.path.join(data_dir, 'images.txt')
        self.prompts_txt = os.path.join(data_dir, 'prompts.txt')

        # Check if the text files exist
        if not os.path.exists(self.images_txt):
            raise FileNotFoundError(f"images.txt not found at {self.images_txt}")
        if not os.path.exists(self.prompts_txt):
            raise FileNotFoundError(f"prompts.txt not found at {self.prompts_txt}")

        # Load image paths and prompts
        self.image_paths = read_text_lines(self.images_txt)
        self.prompts = read_text_lines(self.prompts_txt)

        # Ensure the number of images matches the number of prompts
        assert len(self.image_paths) == len(self.prompts), (
            "The number of images and prompts must be equal."
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, str]: A tuple containing the image tensor and its corresponding prompt.
        """
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Open the image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        prompt = self.prompts[idx]

        # Apply any transformations, if provided
        if self.transform:
            image = self.transform(image)

        # Convert the image to a tensor and normalize it to [-1, 1]
        image = rearrange(
            2 * torch.tensor(np.array(image)).float() / 255 - 1,
            "h w c -> c h w"
        )

        return image, prompt
