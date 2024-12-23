import logging
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all unlearning models."""

    def __init__(self, logger: Any = None):
        super().__init__()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Load the model."""
        pass

    def save_model(self, output_path, *args, **kwargs):
        """
        Save the model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        self.logger.info(f"Saving model to {output_path}...")
        torch.save({"state_dict": self.model.state_dict()}, output_path)
        self.logger.info("Model saved successfully.")
