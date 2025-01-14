# mu/core/base_sampler.py

from abc import ABC, abstractmethod
from typing import Any

class BaseSampler(ABC):
    """Abstract base class for sampling methods used in unlearning."""

    @abstractmethod
    def sample(self, **kwargs) -> Any:
        """Generate samples from the model.

        Args:
            num_samples (int): Number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Generated samples.
        """
        pass

    def load_model(self, *args, **kwargs):
        """Load an image."""
        pass

    def save_image(self, *args, **kwargs):
        """Save an image."""
        pass

