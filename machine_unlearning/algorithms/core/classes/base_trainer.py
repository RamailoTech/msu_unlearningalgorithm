# base_trainer.py

from abc import ABC, abstractmethod
from typing import Any

class BaseTrainer(ABC):
    """Abstract base class for training unlearning models."""

    def __init__(self, model: Any, config: dict, **kwargs):
        self.model = model
        self.config = config


    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def compute_loss(self, output: Any, target: Any) -> Any:
        """Compute the loss for a given output and target."""
        pass

    @abstractmethod
    def step_optimizer(self):
        """Perform a step on the optimizer."""
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        """Validate the model."""
        pass

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        """Save a checkpoint of the model."""
        pass

    @abstractmethod
    def get_model_params(self) -> Any:
        """Get the model parameters."""
        pass

    @abstractmethod
    def set_model_params(self, params: Any):
        """Set the model parameters."""
        pass

