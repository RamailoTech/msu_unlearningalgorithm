# semipermeable_membrane/data_handler.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from algorithms.semipermeable_membrane.src.configs.prompt import PromptSettings
from core.base_data_handler import BaseDataHandler
from datasets.constants import *


class SemipermeableMembraneDataHandler(BaseDataHandler):
    """
    DataHandler for the Semipermeable Membrane algorithm.
    Extends the core DataHandler to generate specific prompts based on themes and classes.
    """


    def __init__(
        self,
        config,
        selected_theme: str,
        selected_class: str,
        use_sample: bool = False
    ):
        """
        Initialize the SemipermeableMembraneDataHandler.

        Args:
            selected_theme (str): The specific theme to edit.
            selected_class (str): The specific class to edit.
            use_sample (bool, optional): Flag to use a sample dataset. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.selected_theme = selected_theme
        self.selected_class = selected_class
        self.use_sample = use_sample

        # Select available themes and classes based on use_sample flag
        if self.use_sample:
            self.theme_available = uc_sample_theme_available
            self.class_available = uc_sample_class_available
            self.logger = logging.getLogger('SemipermeableMembraneDataHandler_Sample')
            self.logger.info("Using sample themes and classes.")
        else:
            self.theme_available = uc_theme_available
            self.class_available = uc_class_available
            self.logger = logging.getLogger('SemipermeableMembraneDataHandler_Full')
            self.logger.info("Using full themes and classes.")


    def load_prompts(self):
        """
        Load prompts from the prompts_file.
        Returns:
            List[PromptSettings]: List of prompt configurations.
        """
        prompts = []
        prompt_dict = self.config.get('prompt')
        prompt = PromptSettings(
            target=prompt_dict.get('target', ''),
            positive=prompt_dict.get('positive', ''),
            unconditional=prompt_dict.get('unconditional', ''),
            neutral=prompt_dict.get('neutral', ''),
            action=prompt_dict.get('action', ''),
            guidance_scale=float(prompt_dict.get('guidance_scale', 1.0)),
            resolution=int(prompt_dict.get('resolution', 512)),
            batch_size=int(prompt_dict.get('batch_size', 1)),
            dynamic_resolution=bool(prompt_dict.get('dynamic_resolution', False)),
            la_strength=int(prompt_dict.get('la_strength', 1000)),
            sampling_batch_size=int(prompt_dict.get('sampling_batch_size', 4))
        )
        prompts.append(prompt)
        self.logger.info(f"Loaded prompt: {prompt.target}")

        return prompts
