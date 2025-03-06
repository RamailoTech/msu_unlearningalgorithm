#mu/algorithms/erase_diff/evaluator.py

import sys
import logging
import timm
import torch

from mu.datasets.constants import *
from evaluation.core import BaseEvaluator
from mu.algorithms.erase_diff import EraseDiffSampler
from mu.algorithms.erase_diff.configs import ErasediffEvaluationConfig

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available



class EraseDiffEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
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
        self.device = self.config['devices'][0]
        self.erase_diff_sampler = None

        self.logger = logging.getLogger(__name__)

    def sampler(self, *args, **kwargs):
        self.erase_diff_sampler = EraseDiffSampler(self.config)

    def load_model(self, *args, **kwargs):
        """
        Load the classification model for evaluation, using 'timm' 
        or any approach you prefer. 
        We assume your config has 'ckpt_path' and 'task' keys, etc.
        """
        self.logger.info("Loading classification model...")
        classification_model = self.config.get("classification_model")
        model = timm.create_model(
            classification_model, 
            pretrained=True
        ).to(self.device)
        task = self.config['task'] # "style" or "class"
        num_classes = len(theme_available) if task == "style" else len(class_available)
        # num_classes = 2 
        model.head = torch.nn.Linear(1024, num_classes).to(self.device)

        # Load checkpoint
        ckpt_path = self.config["classifier_ckpt_path"]
        self.logger.info(f"Loading classification checkpoint from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["model_state_dict"])
        model.eval()
        self.logger.info("Classification model loaded successfully.")
        return model


    def generate_images(self, *args, **kwargs):

        self.sampler()

        self.erase_diff_sampler.load_model()
        
        gen_images_dir = self.erase_diff_sampler.sample()  

        return gen_images_dir #return generated images dir classification model 

