# core/base_data_handler.py

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Optional
from torch.utils.data import DataLoader, Dataset
import os
import torch

class BaseDataHandler(ABC):
    """
    Abstract base class for data handling and processing.
    Defines the interface for loading, preprocessing, and providing data loaders.
    """

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

    @abstractmethod
    def get_data_loaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """
        Get data loaders for various data splits.

        Args:
            batch_size (int): Batch size for data loaders.

        Returns:
            Dict[str, DataLoader]: Dictionary containing data loaders, e.g., {'train': train_loader, 'val': val_loader, ...}
        """
        pass

    def setup(self, **kwargs):
        """
        Optional setup method to initialize any additional configurations or resources.
        Can be overridden by subclasses if needed.
        """
        pass
