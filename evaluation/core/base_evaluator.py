# evaluation/core/mu_base_evaluator.py

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEvaluator(ABC):
    """Abstract base class for evaluating image generation models."""
    
    def __init__(self,config: Dict[str, Any], **kwargs):
        self.config = config
    
    def _parse_config(self):
        """
        Parse the configuration parameters for the algorithm.
        """
        # Parse devices
        devices = [
            f"cuda:{int(d.strip())}" for d in self.config.get("devices", "0").split(",")
        ]
        self.config["devices"] = devices
        

    def load_model(self, *args, **kwargs):
        """Load the model for evaluation."""
        pass
    
    @abstractmethod
    def sampler(self, *args, **kwargs):
        """generate sample images"""
        pass

    def generate_images():
        """
        input:
            config:  Dictionary of hyperparams / settings.
        Returns:
            generated_images_path
        """
        pass