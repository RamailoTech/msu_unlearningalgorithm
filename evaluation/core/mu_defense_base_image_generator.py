# evaluation/core/mu_defense_base_image_generator.py

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseImageGenerator(ABC):
    """Abstract class for image generation for mu_defesne evaluation framework"""

    def __init__(self,config: Dict[str, Any], **kwargs):
        self.config = config

    @abstractmethod
    def generate_images(self):
        """
        Generate images from prompts contained in a CSV file.

        Parameters
        ----------
        prompts_path : str
            Path to the CSV file containing prompts and seeds.
        num_samples : int, optional
            Number of samples to generate per prompt. Default is 1.
        from_case : int, optional
            Starting case number (skip earlier cases). Default is 0.
        """
        pass