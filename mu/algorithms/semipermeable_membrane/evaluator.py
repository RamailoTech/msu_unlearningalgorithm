#mu/algorithms/semipermeable_membrane/evaluator.py


import logging

from mu.datasets.constants import *
from mu.algorithms.semipermeable_membrane.configs import SemipermeableMembraneEvaluationConfig
from evaluation.core import BaseEvaluator

from mu.algorithms.semipermeable_membrane import SemipermeableMembraneSampler



class SemipermeableMembraneEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.

    Lyu, M., Yang, Y., Hong, H., Chen, H., Jin, X., He, Y., Xue, H., Han, J., & Ding, G. (2023).

    One-dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications

    https://arxiv.org/abs/2312.16145
    """

    def __init__(self,config: SemipermeableMembraneEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., SemipermeableMembraneSampler).
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
        self.semipermeable_membrane_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.semipermeable_membrane_sampler = SemipermeableMembraneSampler(self.config)
 
    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.semipermeable_membrane_sampler.load_model()
        
        gen_images_dir = self.semipermeable_membrane_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

