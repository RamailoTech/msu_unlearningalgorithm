from abc import ABC, abstractmethod
from typing import Dict

class BaseImageGenerator(ABC):
    """Abstract base class for all image generators."""

    @abstractmethod
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): Configuration parameters for sampling unlearned models.
        """
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load an image."""
        pass

    @abstractmethod
    def sample_image(self, *args, **kwargs):
        """Generate an image."""
        pass

    @abstractmethod
    def save_image(self, *args, **kwargs):
        """Save an image."""
        pass

    