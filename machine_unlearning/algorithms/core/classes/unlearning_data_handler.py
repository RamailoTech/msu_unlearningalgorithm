from abc import ABC, abstractmethod

class AbstractDataHandler(ABC):
    """Abstract base class for data handling and processing."""

    @abstractmethod
    def load_data(self, data_path, batch_size):
        """Load data from the specified path."""
        pass

    @abstractmethod
    def preprocess_data(self, data):
        """Preprocess the data (e.g., normalization, augmentation)."""
        pass