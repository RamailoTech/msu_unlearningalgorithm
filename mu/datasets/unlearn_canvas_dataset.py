# datasets/unlearn_canvas_dataset.py

import os
from typing import Any, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms

from mu.datasets import BaseDataset
from mu.datasets.constants import *


class UnlearnCanvasDataset(BaseDataset):
    """
    Dataset for UnlearnCanvas algorithm.
    Allows selection of specific themes and classes.
    """

    def __init__(
        self,
        data_dir: str,
        template: str,
        template_name: str,
        use_sample: bool = False,
        transform: Any = None,
    ):
        """
        Initialize the UnlearnCanvasDataset.

        Args:
            data_dir (str): Root directory containing dataset.
            template (str): Template type ('style', 'object', or 'i2p').
            template_name (str): Name of the template to use, which can be a class name or a theme name depending on the template (e.g., 'self-harm', 'Abstractionism').
            use_sample (bool, optional): Whether to use sample constants. Defaults to False.
            transform (Any, optional): Transformations to apply to the images.
        """
        super().__init__()

        if template == "style":
            available_options = (
                uc_sample_theme_available if use_sample else uc_theme_available
            )
        elif template == "object":
            available_options = (
                uc_sample_class_available if use_sample else uc_class_available
            )
        else:
            raise ValueError(f"Unsupported template type: {template}")

        assert (
            template_name in available_options
        ), f"Selected template name '{template_name}' is not available for template type '{template}'."

        self.template = template
        self.template_name = template_name
        self.transform = transform

        # Paths to images and prompts
        self.images_txt = os.path.join(data_dir, "images.txt")
        self.prompts_txt = os.path.join(data_dir, "prompts.txt")

        # Check if files exist
        if not os.path.exists(self.images_txt):
            raise FileNotFoundError(f"images.txt not found at {self.images_txt}")
        if not os.path.exists(self.prompts_txt):
            raise FileNotFoundError(f"prompts.txt not found at {self.prompts_txt}")

        # Load image paths and prompts
        self.image_paths = self.read_text_lines(self.images_txt)
        self.prompts = self.read_text_lines(self.prompts_txt)

        assert len(self.image_paths) == len(
            self.prompts
        ), "Number of images and prompts must be equal."

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
            2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w"
        )

        return image, prompt
