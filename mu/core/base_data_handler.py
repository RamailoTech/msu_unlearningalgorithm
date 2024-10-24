from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseDataHandler(ABC):
    """Abstract base class for data handling and processing."""

    @abstractmethod
    def load_data(self, data_path: str) -> Any:
        """Load data from the specified path.

        Args:
            data_path (str): Path to the data.

        Returns:
            Any: Loaded data.
        """
        pass

    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """Preprocess the data (e.g., normalization, augmentation).

        Args:
            data (Any): Raw data to preprocess.

        Returns:
            Any: Preprocessed data.
        """
        pass

    @abstractmethod
    def get_data_loaders(self, batch_size: int) -> Tuple[Any, Any, Any]:
        """Get data loaders for training, validation, and testing.

        Args:
            batch_size (int): Batch size for data loaders.

        Returns:
            Tuple[Any, Any, Any]: Data loaders for training, validation, and testing.
        """
        pass
