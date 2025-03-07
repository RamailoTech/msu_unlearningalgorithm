#mu/algorithms/unified_concept_editing/evaluator.py
import logging

from mu.datasets.constants import *
from mu.algorithms.unified_concept_editing.configs import UceEvaluationConfig
from mu.algorithms.unified_concept_editing import UnifiedConceptEditingSampler

from evaluation.core import BaseEvaluator

class UnifiedConceptEditingEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.

    Gandikota, R., Orgad, H., Belinkov, Y., Materzyńska, J., & Bau, D. (2023).

    Unified Concept Editing in Diffusion Models

    https://arxiv.org/abs/2308.14761
    """

    def __init__(self,config:UceEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., UnifiedConceptEditingSampler).
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
        self.uce_sampler = None

        self.results = {}

        self.logger = logging.getLogger(__name__)


    def sampler(self, *args, **kwargs):
        self.uce_sampler = UnifiedConceptEditingSampler(self.config)
 
    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.uce_sampler.load_model()
        
        gen_images_dir = self.uce_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 
