# mu/core/base_trainer.py

from abc import ABC, abstractmethod
from typing import Any

class BaseTrainer(ABC):
    """Abstract base class for training unlearning models."""

    def __init__(self, model: Any, config: dict, **kwargs):
        self.model = model
        self.config = config


    # @abstractmethod
    def setup_optimizer(self, *args, **kwargs):
        """Set up the optimizers for training."""
        pass

    # @abstractmethod
    def train(self, *args, **kwargs):
        """Train the model."""
        pass

