from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any

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


    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """Perform a forward pass on the input data.

        Args:
            input_data (Any): Input data for the model.

        Returns:
            Any: Model output.
        """
        pass