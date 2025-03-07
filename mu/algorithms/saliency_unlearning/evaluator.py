#mu/algorithms/saliency_unlearning/evaluator.py

import sys
import os
import logging
import torch
import timm
import json


from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available
from mu.datasets.constants import *
from evaluation.core import BaseEvaluator
from mu.algorithms.saliency_unlearning import SaliencyUnlearningSampler
from mu.algorithms.saliency_unlearning.configs import SaliencyUnlearningEvaluationConfig



class SaliencyUnlearningEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.

    Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D., & Liu, S. (2023).

    SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation

    https://arxiv.org/abs/2310.12508
    """

    def __init__(self,config:SaliencyUnlearningEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., SaliencyUnlearningSampler).
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
        self.device = self.config['devices'][0]
        self.use_sample = self.config.get('use_sample')
        self.dataset_type = self.config.get('dataset_type')
        self.saliency_unlearning_sampler = None
        self.results = {}

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.saliency_unlearning_sampler = SaliencyUnlearningSampler(self.config)


    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.saliency_unlearning_sampler.load_model()
        
        gen_images_dir = self.saliency_unlearning_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

