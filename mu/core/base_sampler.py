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
