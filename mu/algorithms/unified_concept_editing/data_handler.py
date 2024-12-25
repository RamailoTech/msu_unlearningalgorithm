# unified_concept_editing/data_handler.py

from typing import List, Optional, Tuple
import logging

from mu.core import BaseDataHandler
from mu.datasets.constants import *

class UnifiedConceptEditingDataHandler(BaseDataHandler):
    """
    DataHandler for Unified Concept Editing.
    Extends the core DataHandler to generate specific prompts based on themes and classes.
    """

    def __init__(
        self,
        dataset_type: str,
        template: str,
        template_name: str,
        use_sample: bool = False
    ):
        """
        Initialize the UnifiedConceptEditingDataHandler.

        Args:
            dataset_type (str, optional): Type of dataset to use. Defaults to 'unlearncanvas'.
            template (str): Template type ('style' or 'object')
            template_name (str): Name of the template to use
            use_sample (bool, optional): Flag to use a sample dataset. Defaults to False.
        """
        super().__init__()
        self.dataset_type = dataset_type
        self.template = template
        self.template_name = template_name
        self.use_sample = use_sample

        self.logger = logging.getLogger(__name__)

        if self.dataset_type == 'unlearncanvas':
            if template == 'style':
                self.concepts_available = uc_sample_theme_available if self.use_sample else uc_theme_available
            elif template == 'object' :
                self.concepts_available = uc_sample_class_available if self.use_sample else uc_class_available

        elif self.dataset_type == 'i2p':
            self.concepts_available = i2p_sample_categories if self.use_sample else i2p_categories

    def generate_dataset(
        self,
        add_prompts: bool,
        guided_concepts: Optional[str],
        preserve_concepts: Optional[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generate old_texts, new_texts, and retain_texts based on the selected theme, class, and configuration.

        Args:
            add_prompts (bool): Whether to add additional prompts.
            guided_concepts (Optional[str]): Comma-separated string of concepts to guide the editing.
            preserve_concepts (Optional[str]): Comma-separated string of concepts to preserve.

        Returns:
            Tuple[List[str], List[str], List[str]]: Lists of old texts, new texts, and retain texts.
        """
        old_texts = []
        new_texts = []
        retain_texts = []

        additional_prompts = []
        concept = self.template_name
        if self.template == 'style' and self.template_name in self.concepts_available:
            additional_prompts = [
                'image in {concept} Style',
                'art by {concept}',
                'artwork by {concept}',
                'picture by {concept}',
                'style of {concept}'
            ]
        elif self.template == 'object' and self.template_name in self.concepts_available:
            additional_prompts = [
                'image of {concept}',
                'photo of {concept}',
                'portrait of {concept}',
                'picture of {concept}',
                'painting of {concept}'
            ]

        if not add_prompts:
            additional_prompts = []
 
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))

        # Prepare new_texts based on guided_concepts
        if guided_concepts is None:
            new_texts = [' ' for _ in old_texts]
        else:
            guided_concepts = [con.strip() for con in guided_concepts.split(',')]
            if len(guided_concepts) == 1:
                new_texts = [guided_concepts[0] for _ in old_texts]
            else:
                new_texts = []
                for con in guided_concepts:
                    new_texts.extend([con] * (1 + len(additional_prompts)))

        assert len(new_texts) == len(old_texts), "Length of new_texts must match old_texts."

        # Prepare retain_texts based on preserve_concepts
        if preserve_concepts is None:
            retain_texts = ['']
        else:
            preserve_concepts = [con.strip() for con in preserve_concepts.split(',')]
            for con in preserve_concepts:
                for theme_item in self.concepts_available:
                    if theme_item == "Seed_Images":
                        adjusted_theme = "Photo"
                    else:
                        adjusted_theme = theme_item
                    retain_texts.append(f'A {con} image in {adjusted_theme} style')

        self.logger.info(f"Old Texts: {old_texts}")
        self.logger.info(f"New Texts: {new_texts}")
        self.logger.info(f"Retain Texts: {retain_texts}")

        return old_texts, new_texts, retain_texts

    def load_data(self, data_path: str):
        """
        Load data from the specified path.

        Args:
            data_path (str): Path to the data.

        Returns:
            Any: Loaded data.
        """
        pass

    def preprocess_data(self, data):
        """
        Preprocess the data (e.g., normalization, augmentation).

        Args:
            data (Any): Raw data to preprocess.

        Returns:
            Any: Preprocessed data.
        """
        pass
    