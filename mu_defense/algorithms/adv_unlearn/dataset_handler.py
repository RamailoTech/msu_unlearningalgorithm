
# mu_defense/algorithms/adv_unlearn/dataset_handler.py

import logging

from mu_defense.core import BaseDatasetHandler
from mu_defense.algorithms.adv_unlearn.utils import retain_prompt  


class AdvUnlearnDatasetHandler(BaseDatasetHandler):
    """
    AdvUnlearnDatasetHandler handles prompt cleaning and retaining dataset creation
    specifically for adversarial unlearning.
    """
    def __init__(self,prompt, seperator, dataset_retain):
        self.prompt = prompt
        self.seperator = seperator
        self.dataset_retain = dataset_retain
        # self.use_sample = use_sample

        self.logger = logging.getLogger(__name__)


    def setup_prompt(self):
        # Special-case prompt cleaning.
        if self.prompt == 'allartist':
            self.prompt = ("Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, "
                           "Alena Aenami, Tyler Edlin, Kilian Eng")
        elif self.prompt == 'i2p':
            self.prompt = ("hate, harassment, violence, suffering, humiliation, harm, "
                           "suicide, sexual, nudity, bodily fluids, blood")
        elif self.prompt == "artifact":
            self.prompt = ("ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
                           "mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, "
                           "body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy")
        
        if self.seperator:
            self.words = [w.strip() for w in self.prompt.split(self.seperator)]
        else:
            self.words = [self.prompt]
        
        self.word_print = self.prompt.replace(" ", "")
        self.logger.info(f"The Concept Prompt to be unlearned: {self.words}")
        return self.words, self.word_print

    def setup_dataset(self):
        """
        Create and return the retaining dataset using the helper function.
        """
        dataset = retain_prompt(self.dataset_retain)
        return dataset
