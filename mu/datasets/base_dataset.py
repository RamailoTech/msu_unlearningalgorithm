from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch.utils.data import Dataset


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

    def read_text_lines(self, path: str):
        """
        Read lines from a text file.

        Args:
            path (str): Path to the text file.

        Returns:
            List[str]: List of lines.
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
