# mu_defense/algorithms/adv_unlearn/evaluator.py

import logging

from evaluation.core import BaseEvaluator

from mu_defense.algorithms.adv_unlearn.configs import MUDefenseEvaluationConfig
from mu_defense.algorithms.adv_unlearn.image_generator import ImageGenerator

class MUDefenseEvaluator(BaseEvaluator):
    """Evaluator for the defense."""
    
    def __init__(self, config: MUDefenseEvaluationConfig,**kwargs):
        """Initialize the evaluator."""
        super().__init__(config, **kwargs)
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        self.image_generator = None
        self.logger = logging.getLogger(__name__)

        # self._parse_config()
        config.validate_config()

    def sampler(self):
        self.image_generator = ImageGenerator(self.config)

    def generate_images(self):
        self.sampler()
        gen_img_path = self.image_generator.generate_images()
        return gen_img_path

    
   