# mu/algorithms/forget_me_not/evaluator.py

import logging

from evaluation.core import BaseEvaluator
from mu.datasets.constants import *

from mu.algorithms.forget_me_not.configs import ForgetMeNotEvaluationConfig
from mu.algorithms.forget_me_not import ForgetMeNotSampler


class ForgetMeNotEvaluator(BaseEvaluator):

    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.


    Zhang, E., Wang, K., Xu, X., Wang, Z., & Shi, H. (2023).

    Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models

    https://arxiv.org/abs/2211.08332
    """

    def __init__(self,config: ForgetMeNotEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., ForgetMeNotSampler).
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

        self.forget_me_not_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.forget_me_not_sampler = ForgetMeNotSampler(self.config)


    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.forget_me_not_sampler.load_model()
        
        gen_images_dir = self.forget_me_not_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 
