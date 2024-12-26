# mu/core/base_data_handler.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset


class BaseDataHandler(ABC):
    """
    Abstract base class for data handling and processing.
    Defines the interface for loading, preprocessing, and providing data loaders.
    """
    @abstractmethod
    def generate_dataset(self):
        """
        Generate the dataset.
        """
        pass

    @abstractmethod
    def load_data(self, data_path: str) -> Any:
        """
        Load data from the specified path.

        Args:
            data_path (str): Path to the data.

        Returns:
            Any: Loaded data.
        """
        pass

    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """
        Preprocess the data (e.g., normalization, augmentation).

        Args:
            data (Any): Raw data to preprocess.

        Returns:
            Any: Preprocessed data.
        """
        pass

