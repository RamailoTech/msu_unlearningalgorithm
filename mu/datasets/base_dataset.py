from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Any, Tuple

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    """
    @abstractmethod
    def __init__(self):
        """
        Initialize the dataset.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, str]: A tuple containing the data sample and its corresponding prompt.
        """
        pass
