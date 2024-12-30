# mu/algorithms/semipermeable_membrane/data_handler.py

from typing import Any, Dict
import logging

from mu.core import BaseDataHandler
from mu.algorithms.semipermeable_membrane.src.configs.prompt import PromptSettings

class SemipermeableMembraneDataHandler(BaseDataHandler):
    """
    DataHandler for the Semipermeable Membrane algorithm.
    Extends the core DataHandler to generate specific prompts based on themes and classes.
    """

    def __init__(
        self,
        template: str,
        template_name: str,
        dataset_type: str = 'unlearncanvas',
        use_sample: bool = False,
        config: Dict = None,
    ):
        """
        Initialize the SemipermeableMembraneDataHandler.

        Args:
            template (str): Template type ('style' or 'object')
            template_name (str): Name of the template to use
            dataset_type (str, optional): Type of dataset to use. Defaults to 'unlearncanvas'.
            use_sample (bool, optional): Flag to use a sample dataset. Defaults to False.
            config (Dict, optional): Configuration dictionary. Defaults to None.
        """
        super().__init__()
        self.template_name = template_name
        self.template = template
        self.dataset_type = dataset_type
        self.use_sample = use_sample
        self.config = config 
        self.logger = logging.getLogger(__name__)

    def generate_dataset(self, *args, **kwargs):
        pass 

    def preprocess_data(self, *args, **kwargs):
        pass 

    def load_data(self, *args, **kwargs):
        """
        Load prompts from the prompts_file.
        Returns:
            List[PromptSettings]: List of prompt configurations.
        """
        prompts = []
        # Safely get the prompt dictionary from the config
        prompt_dict = getattr(self.config, 'prompt', {})

        # Create a PromptSettings object with safe access to prompt attributes
        prompt = PromptSettings(
            target=prompt_dict.get('target', ''),
            positive=prompt_dict.get('positive', ''),
            unconditional=prompt_dict.get('unconditional', ''),
            neutral=prompt_dict.get('neutral', ''),
            action=prompt_dict.get('action', 'erase'),
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
