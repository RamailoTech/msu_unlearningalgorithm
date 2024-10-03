from abc import ABC, abstractmethod


class UnlearningTrainer(ABC):
    """Abstract base class for training unlearning models."""

    @abstractmethod
    def train(self, model, sampler, **kwargs):
        """Train the model with the specified optimizer and number of epochs."""
        pass

    @abstractmethod
    def compute_loss(self, output, target):
        """Compute the loss for a given output and target."""
        pass

    @abstractmethod
    def step_optimizer(self, optimizer):
        """Perform a step on the optimizer."""
        pass