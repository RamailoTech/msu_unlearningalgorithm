from abc import ABC, abstractmethod

class AbstractSampler(ABC):
    """Abstract base class for sampling methods used in unlearning."""

    @abstractmethod
    def sample(self, **kwargs):
        """Generate samples from the model."""
        pass