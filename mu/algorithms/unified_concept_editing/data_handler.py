# unified_concept_editing/data_handler.py

from mu.core.base_data_handler import BaseDataHandler
from typing import List, Optional, Tuple
import logging
from mu.datasets.constants import *


class UnifiedConceptEditingDataHandler(BaseDataHandler):
    """
    DataHandler for Unified Concept Editing.
    Extends the core DataHandler to generate specific prompts based on themes and classes.
    """

    def __init__(
        self, selected_theme: str, selected_class: str, use_sample: bool = False
    ):
        """
        Initialize the UnifiedConceptEditingDataHandler.

        Args:
            original_data_dir (str): Directory containing the original dataset organized by themes and classes.
            new_data_dir (str): Directory where the new datasets will be saved.
            selected_theme (str): The specific theme to edit.
            selected_class (str): The specific class to edit.
            use_sample (bool, optional): Flag to use a sample dataset. Defaults to False.
        """
        super().__init__()
        self.selected_theme = selected_theme
        self.selected_class = selected_class
        self.use_sample = use_sample

        # Select available themes and classes based on use_sample flag
        if self.use_sample:
            self.theme_available = uc_sample_theme_available
            self.class_available = uc_sample_class_available
            self.logger = logging.getLogger("UnifiedConceptEditingDataHandler_Sample")
            self.logger.info("Using sample themes and classes.")
        else:
            self.theme_available = uc_theme_available
            self.class_available = uc_class_available
            self.logger = logging.getLogger("UnifiedConceptEditingDataHandler_Full")
            self.logger.info("Using full themes and classes.")

    def prepare_prompts(
        self,
        add_prompts: bool,
        guided_concepts: Optional[str],
        preserve_concepts: Optional[str],
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
        theme = self.selected_theme

        # Determine additional prompts based on the selected theme
        if theme in self.theme_available:
            additional_prompts = [
                "image in {concept} Style",
                "art by {concept}",
                "artwork by {concept}",
                "picture by {concept}",
                "style of {concept}",
            ]
        elif theme in self.class_available:
            additional_prompts = [
                "image of {concept}",
                "photo of {concept}",
                "portrait of {concept}",
                "picture of {concept}",
                "painting of {concept}",
            ]

        if not add_prompts:
            additional_prompts = []

        concepts = [theme]
        for concept in concepts:
            old_texts.append(f"{concept}")
            for prompt in additional_prompts:
                old_texts.append(prompt.format(concept=concept))

        # Prepare new_texts based on guided_concepts
        if guided_concepts is None:
            new_texts = [" " for _ in old_texts]
        else:
            guided_concepts = [con.strip() for con in guided_concepts.split(",")]
            if len(guided_concepts) == 1:
                new_texts = [guided_concepts[0] for _ in old_texts]
            else:
                new_texts = []
                for con in guided_concepts:
                    new_texts.extend([con] * (1 + len(additional_prompts)))

        assert len(new_texts) == len(
            old_texts
        ), "Length of new_texts must match old_texts."

        # Prepare retain_texts based on preserve_concepts
        if preserve_concepts is None:
            retain_texts = [""]
        else:
            preserve_concepts = [con.strip() for con in preserve_concepts.split(",")]
            for con in preserve_concepts:
                for theme_item in self.theme_available:
                    if theme_item == "Seed_Images":
                        adjusted_theme = "Photo"
                    else:
                        adjusted_theme = theme_item
                    retain_texts.append(f"A {con} image in {adjusted_theme} style")

        self.logger.info(f"Old Texts: {old_texts}")
        self.logger.info(f"New Texts: {new_texts}")
        self.logger.info(f"Retain Texts: {retain_texts}")

        return old_texts, new_texts, retain_texts
