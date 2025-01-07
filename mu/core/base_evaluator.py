from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEvaluator(ABC):
    """Abstract base class for evaluating image generation models."""
    
    def __init__(self,config: Dict[str, Any], **kwargs):
        # self.sampler =sampler
        self.config = config

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load the model for evaluation."""
        pass

    @abstractmethod
    def preprocess_image(self, *args, **kwargs):
        """Preprocess images before evaluation."""
        pass

    @abstractmethod
    def calculate_accuracy(self, *args, **kwargs):
        """Calculate accuracy of the model."""
        pass

    @abstractmethod
    def calculate_fid_score(self, *args, **kwargs):
        """Calculate the Fréchet Inception Distance (FID) score."""
        pass

    @abstractmethod
    def save_results(self, *args, **kwargs):
        """Save evaluation results to a file."""
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the evaluation process."""
        pass