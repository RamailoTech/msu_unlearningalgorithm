# mu/core/base_model.py

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Abstract base class for all unlearning models."""

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load the model."""
        pass

    @abstractmethod
    def save_model(self, *args, **kwargs):
        """Save the model."""
        pass
