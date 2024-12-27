# mu/core/base_data_handler.py

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional, List
from torch.utils.data import DataLoader, Dataset

class BaseDataHandler(ABC):
    """
    Abstract base class for data handling and processing.
    Defines the interface for loading, preprocessing, and providing data loaders.
    """

    @abstractmethod
    def generate_dataset(self, *args, **kwargs) -> Any:
        """
        Generate the dataset.
        """
        pass

    
    @abstractmethod
    def load_data(self, *args, **kwargs) -> Any:
        """
        Load data

        Returns:
            Any: Loaded data.
        """
        pass

    @abstractmethod
    def preprocess_data(self, *args, **kwargs ) -> Any:
        """
        Preprocess the data (e.g., normalization, augmentation).

        Returns:
            Any: Preprocessed data.
        """
        pass

