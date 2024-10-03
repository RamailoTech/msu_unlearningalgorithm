from abc import ABC, abstractmethod

class UnlearningModel(ABC):
    """Abstract base class for all unlearning models."""

    @abstractmethod
    def load_model(self, config_path, ckpt_path, device):
        """Load the model using configuration and checkpoint."""
        pass

    @abstractmethod
    def save_model(self, output_path):
        """Save the trained model to the output path."""
        pass

    @abstractmethod
    def forward_pass(self, input_data):
        """Perform a forward pass on the input data."""
        pass