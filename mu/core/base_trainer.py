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
        pass

    @abstractmethod
    def compute_loss(self, output: Any, target: Any) -> Any:
        pass

    @abstractmethod
    def step_optimizer(self):
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_model_params(self) -> Any:
        pass

    @abstractmethod
    def set_model_params(self, params: Any):
        pass

