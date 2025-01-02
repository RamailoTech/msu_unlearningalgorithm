from abc import ABC, abstractmethod
import os
from PIL import Image
import torch
from pytorch_lightning import seed_everything


class ImageGenerator(ABC):
    """
    Abstract base class for generating images using deep learning models.

    This class provides a structured interface for image generation tasks. It handles
    model loading, seed setup for reproducibility, and saving generated images.
    Subclasses must implement the `load_model` and `generate_images` methods to define
    model-specific behavior.
    """

    @abstractmethod
    def load_model(self):
        """
        Abstract method for loading the model.
        """
        pass

    @abstractmethod
    def generate_images(self, themes, classes, seeds):
        """
        Abstract method for generating images.

        Subclasses must implement this method to define how images are generated
        using the model, including the process of sampling and saving.

        Args:
            themes (list[str]): List of themes (styles) for image generation.
            classes (list[str]): List of object classes to generate.
            seeds (list[int]): List of seeds for random initialization.
        """
        pass

    @abstractmethod
    def save_image(self, image_tensor, file_path):
        """i
        Save an image
        """
        pass

    @staticmethod
    def setup_seed(seed):
        """
        Set up the random seed for reproducibility.

        Ensures that random operations in PyTorch and other libraries are deterministic
        for a given seed value.

        Args:
            seed (int): The seed value to set.
        """
        seed_everything(seed)

    def generate_prompt(self, theme, object_class):
        """
        Generate a text prompt based on the theme and object class.

        Args:
            theme (str): Theme or style for the image.
            object_class (str): Object class to be included in the image.

        Returns:
            str: A formatted text prompt.
        """
        return f"A {object_class} image in {theme} style."
