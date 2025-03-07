#mu/algorithms/concept_ablation/evaluator.py

import sys
import os
import logging
import torch

from mu.algorithms.concept_ablation import ConceptAblationSampler
from mu.datasets.constants import *
from mu.algorithms.concept_ablation.configs import ConceptAblationEvaluationConfig
from evaluation.core import BaseEvaluator



class ConceptAblationEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.


    Kumari, N., Zhang, B., Wang, S.-Y., Shechtman, E., Zhang, R., & Zhu, J.-Y. (2023).

    Ablating Concepts in Text-to-Image Diffusion Models

    Presented at the 2023 IEEE International Conference on Computer Vision
    """

    def __init__(self,config: ConceptAblationEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., ConceptAblationSampler).
            config (Dict[str, Any]): A dict of hyperparameters / evaluation settings.
            **kwargs: Additional overrides for config.
        """
        super().__init__(config, **kwargs)
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value) 
        self._parse_config()
        config.validate_config()
        self.config = config.to_dict()
        self.concept_ablation_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.concept_ablation_sampler = ConceptAblationSampler(self.config)


    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.concept_ablation_sampler.load_model()
        
        gen_images_dir = self.concept_ablation_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

