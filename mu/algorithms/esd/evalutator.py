
#mu/algorithms/esd/evaluator.py

import sys
import os
import logging
import timm

import torch

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available
from mu.datasets.constants import *
from mu.algorithms.esd.configs import ESDEvaluationConfig
from mu.algorithms.esd import ESDEvaluatorSampler
from evaluation.core import BaseEvaluator


class ESDEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.

    Gandikota, R., Materzyńska, J., Fiotto-Kaufman, J., & Bau, D. (2023).

    Erasing Concepts from Diffusion Models

    Presented at the 2023 IEEE International Conference on Computer Vision
    """

    def __init__(self,config: ESDEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., ESDSampler).
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
        self.esd_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.esd_sampler = ESDEvaluatorSampler(self.config)


    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.esd_sampler.load_model()
        
        gen_images_dir = self.esd_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

