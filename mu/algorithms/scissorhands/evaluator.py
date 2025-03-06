# mu/algorithms/scissorhands/evaluator.py

import logging


from mu.datasets.constants import *
from mu.algorithms.scissorhands.configs import ScissorhandsEvaluationConfig
from evaluation.core import BaseEvaluator
from mu.algorithms.scissorhands import ScissorHandsSampler


class ScissorHandsEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.

    Wu, J., & Harandi, M. (2024).

    Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks

    https://arxiv.org/abs/2401.06187
    """

    def __init__(self,config: ScissorhandsEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., ScissorHandsSampler).
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
        self.scissor_hands_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.scissor_hands_sampler = ScissorHandsSampler(self.config)

    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.scissor_hands_sampler.load_model()
        
        gen_images_dir = self.scissor_hands_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

