#mu/algorithms/erase_diff/evaluator.py

import sys
import logging
import torch

from mu.datasets.constants import *
from evaluation.core import BaseEvaluator
from mu.algorithms.erase_diff import EraseDiffSampler
from mu.algorithms.erase_diff.configs import ErasediffEvaluationConfig



class EraseDiffEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.

    Wu, J., Le, T., Hayat, M., & Harandi, M. (2024).

    EraseDiff: Erasing Data Influence in Diffusion Models

    https://arxiv.org/abs/2401.05779
    """

    def __init__(self,config: ErasediffEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., EraseDiffSampler).
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
        self.erase_diff_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.erase_diff_sampler = EraseDiffSampler(self.config)

    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.erase_diff_sampler.load_model()
        
        gen_images_dir = self.erase_diff_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

