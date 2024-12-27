# mu/algorithms/unified_concept_editing/trainer.py

import logging
from typing import Optional

from mu.algorithms.unified_concept_editing.model import UnifiedConceptEditingModel
from mu.algorithms.unified_concept_editing.data_handler import UnifiedConceptEditingDataHandler
from mu.core import BaseTrainer


class UnifiedConceptEditingTrainer(BaseTrainer):
    """
    Trainer for the UnifiedConceptEditing algorithm.
    Handles the model editing process to unify or erase specific concepts within the model.
    """

    def __init__(
        self,
        model: UnifiedConceptEditingModel,
        config: dict,
        device: str,
        data_handler: UnifiedConceptEditingDataHandler,
        **kwargs
    ):
        """
        Initialize the UnifiedConceptEditingTrainer.

        Args:
            model (UnifiedConceptEditingModel): Instance of UnifiedConceptEditingModel.
            config (dict): Configuration dictionary.
            device (str): Device to perform training on.
            data_handler (UnifiedConceptEditingDataHandler): Instance of UnifiedConceptEditingDataHandler.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model, config, **kwargs)
        self.device = device
        self.model = model
        self.data_handler = data_handler
        self.logger = logging.getLogger(__name__)
        self.technique = self.config.get('technique', 'replace')
        self.lamb = float(self.config.get('lamb', 0.5))
        self.erase_scale = self.config.get('erase_scale', 1.0)
        self.preserve_scale = self.config.get('preserve_scale', 0.1)
        self.layers_to_edit = self.config.get('layers_to_edit', None)  # Optional: List[int]
        self.logger.info("Trainer initialized with configuration.")

    def train(self):
        """
        Execute the model editing process.
        """
        # Extract concepts and prompts from data handler
        add_prompts = self.config.get('add_prompts', False)
        guided_concepts = self.config.get('guided_concepts')
        preserve_concepts = self.config.get('preserve_concepts')

        # Prepare prompts using data handler
        old_texts, new_texts, retain_texts = self.data_handler.generate_dataset(
            add_prompts=add_prompts,
            guided_concepts=guided_concepts,
            preserve_concepts=preserve_concepts
        )

        # Perform model editing using the edit_model method from model.py
        pretrained_model = self.model.edit_model(
            old_texts=old_texts,
            new_texts=new_texts,
            retain_texts=retain_texts,
            lamb=self.lamb,
            erase_scale=self.erase_scale,
            preserve_scale=self.preserve_scale,
            layers_to_edit=self.layers_to_edit,
            technique=self.technique
        )

        return pretrained_model