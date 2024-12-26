# mu/algorithms/unified_concept_editing/trainer.py

import logging
import torch
import pandas as pd 
from diffusers import StableDiffusionPipeline



from mu.algorithms.unified_concept_editing.model import UnifiedConceptEditingModel
from mu.algorithms.unified_concept_editing.data_handler import (
    UnifiedConceptEditingDataHandler,
)
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
        self.technique = self.config.get("technique", "replace")
        self.lamb = self.config.get("lamb", 0.5)
        self.erase_scale = self.config.get("erase_scale", 1.0)
        self.preserve_scale = self.config.get("preserve_scale", 0.1)
        self.layers_to_edit = self.config.get(
            "layers_to_edit", None
        )  # Optional: List[int]
        self.logger.info("Trainer initialized with configuration.")
    
    def train(self, mode=None):
        """
        Execute the model editing process.
        """
        assert (
            mode is None
        ), "UnifiedConceptEditingTrainer does not support this training mode."

        if mode=="erase":
            return self.train_erase()
        elif mode=="debias":
            return self.train_debias()

    
    def train_erase(self):
        """
        Execute the model editing process for erasing concepts.
        """
        # Extract concepts and prompts from data handler
        add_prompts = self.config.get("add_prompts", False)
        guided_concepts = self.config.get("guided_concepts")
        preserve_concepts = self.config.get("preserve_concepts")

        # Prepare prompts using data handler
        old_texts, new_texts, retain_texts = self.data_handler.generate_dataset(
            add_prompts=add_prompts,
            guided_concepts=guided_concepts,
            preserve_concepts=preserve_concepts,
        )
        pretrained_model = self.model.edit_model_erase(
            old_texts=old_texts,
            new_texts=new_texts,
            retain_texts=retain_texts,
            lamb=self.lamb,
            erase_scale=self.erase_scale,
            layers_to_edit=self.layers_to_edit,
            )

        return pretrained_model
    
    def train_debias(self):
        """
        Execute the model editing process for debiasing concepts.
        """
        num_images = self.config.get("num_images", 10)
        concepts = self.config.get("concept", "").split(',')
        concepts = [con.strip() for con in concepts]
        print_text = '_'.join([txt.lower() for txt in concepts])

        if concepts[0] == 'default0':
            self.logger.info('Using default training setup')
            concepts = [
                'attendant', 'cashier', 'teacher', 'nurse', 'assistant', 'secretary', 'cleaner', 'receptionist', 'clerk', 'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 'librarian', 'tailor', 'driver', 'supervisor', 'janitor', 'cook', 'laborer', 'construction worker', 'developer', 'carpenter', 'manager', 'lawyer', 'farmer', 'salesperson', 'physician', 'guard', 'analyst', 'mechanic', 'sheriff', 'CEO', 'doctor', 'chef'
            ]

        old_texts = []
        concepts_ = []
        for concept in concepts:
            old_texts.extend([
                f'image of {concept}', f'photo of {concept}', f'portrait of {concept}', f'picture of {concept}', f'{concept}'
            ])
            concepts_.extend([concept] * 5)

        attributes = self.config.get("attributes", "").split(',')
        attributes = [att.strip() for att in attributes]

        print_text += '-attributes-' + '_'.join([txt.lower().replace(' ', '9') for txt in attributes])

        new_texts = [[text.replace(concepts_[idx], att) for att in attributes] for idx, text in enumerate(old_texts)]

        df = pd.read_csv('data/profession_prompts.csv')
        retain_texts = list(df.profession.unique())

        old_texts_lower = [text.lower() for text in old_texts]
        retain_texts = [text for text in retain_texts if text.lower() not in old_texts_lower]

        base_version = self.config.get("base", "1.4")
        model_version = "CompVis/stable-diffusion-v1-4" if base_version == '1.4' else 'stabilityai/stable-diffusion-2-1-base'
        
        ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(self.device)
        print_text += f"-sd_{base_version.replace('.', '_')}"
        self.logger.info(print_text)

        pretrained_model, weights, init_ratios, final_ratios = self.model.edit_model_debias(
            old_text_=old_texts,
            new_text_=new_texts,
            add=False,
            retain_text_=retain_texts,
            lamb=self.lamb,
            erase_scale=self.erase_scale,
            preserve_scale=self.preserve_scale,
            num_images=num_images
        )

        torch.save(ldm_stable.unet.state_dict(), f'models/unbiased-{print_text}.pt')

        with open(f'data/unbiased-{print_text}.txt', 'w') as fp:
            fp.write(f"{old_texts}\n{weights}\n{init_ratios}\n{final_ratios}")  
        
        return pretrained_model