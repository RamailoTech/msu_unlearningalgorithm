# mu/algorithms/forget_me_not/evaluator.py

import sys
import os
import logging
import torch
import timm
import json

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available
from evaluation.core import BaseEvaluator
from mu.datasets.constants import *
from evaluation.evaluators.accuracy import calculate_accuracy_for_dataset
from evaluation.evaluators.fid import calculate_fid_score
from evaluation.evaluators import load_style_generated_images,load_style_ref_images,calculate_fid,tensor_to_float
from mu.algorithms.forget_me_not.configs import ForgetMeNotEvaluationConfig
from mu.algorithms.forget_me_not import ForgetMeNotSampler


class ForgetMeNotEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
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
        self.device = self.config['devices'][0]
        self.use_sample = self.config.get('use_sample')
        self.dataset_type = self.config.get('dataset_type')
        self.sampler = ForgetMeNotSampler(self.config)
        self.model = None
        self.eval_output_path = None
        self.results = {}

        self.logger = logging.getLogger(__name__)


    def load_model(self, *args, **kwargs):
        """
        Load the classification model for evaluation, using 'timm' 
        or any approach you prefer. 
        We assume your config has 'ckpt_path' and 'task' keys, etc.
        """
        self.logger.info("Loading classification model...")
        model = self.config.get("classification_model")
        self.model = timm.create_model(
            model, 
            pretrained=True
        ).to(self.device)
        task = self.config['task'] # "style" or "class"
        num_classes = len(theme_available) if task == "style" else len(class_available)
        self.model.head = torch.nn.Linear(1024, num_classes).to(self.device)

        # Load checkpoint
        ckpt_path = self.config["classifier_ckpt_path"]
        self.logger.info(f"Loading classification checkpoint from: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["model_state_dict"])
        self.model.eval()
    
        self.logger.info("Classification model loaded successfully.")


    def calculate_accuracy(self, *args, **kwargs):
        """
        Calculate accuracy of the classification model on generated images.
        This method now supports both the original datasets and the i2p dataset.
        For the i2p dataset, the config should specify task="i2p" and provide a list of categories.
        """
        self.logger.info("Starting accuracy calculation...")
        self.results = calculate_accuracy_for_dataset(self.config, self.model)
        
        self.logger.info(f"Accuracy: ", self.results)

    def calculate_fid_score(self, *args, **kwargs):
        """
        Calculate the Fréchet Inception Distance (FID) score using the images 
        generated by EraseDiffSampler vs. some reference images. 
        """
        self.logger.info("Starting FID calculation...")
        # self.theme_available = uc_sample_theme_available_eval if self.use_sample else uc_theme_available
        # self.class_available = uc_sample_class_available_eval if self.use_sample else uc_class_available
        generated_path = self.config["sampler_output_dir"]  
        
        if self.dataset_type in ("i2p", "generic"):
            reference_path = f"{self.config['reference_dir']}/images"
        elif self.dataset_type == "unlearncanvas":
            reference_path = self.config['reference_dir']

        fid_value, _ = calculate_fid_score(generated_path, reference_path)  
        self.results["FID"] = fid_value


    def save_results(self, *args, **kwargs):
        """
        Save whatever is present in `self.results` to a JSON file.
        """
        try:
            # Convert all tensors before saving
            converted_results = tensor_to_float(self.results)
            with open(self.eval_output_path, 'w') as json_file:
                json.dump(converted_results, json_file, indent=4)
            self.logger.info(f"Results saved to: {self.eval_output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to JSON file: {e}")

    def run(self, *args, **kwargs):
        """
       Run the complete evaluation process:
        1) Load the model checkpoint
        2) Generate images (using sampler)
        3) Load the classification model
        4) Calculate accuracy
        5) Calculate FID
        6) Save final results
        """

        # Call the sample method to generate images
        self.sampler.load_model()  
        self.sampler.sample()    

        # Load the classification model
        self.load_model()

        # Proceed with accuracy and FID calculations
        self.calculate_accuracy()
        self.calculate_fid_score()

        # Save results
        self.save_results()

        self.logger.info("Evaluation run completed.")

