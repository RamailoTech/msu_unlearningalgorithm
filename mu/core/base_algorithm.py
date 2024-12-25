# mu/core/base_algorithm.py

from abc import ABC, abstractmethod
from typing import Dict

class BaseAlgorithm(ABC):
    """
    Abstract base class for the overall unlearning algorithm, combining the model, trainer, and sampler.
    All algorithms must inherit from this class and implement its methods.
    """

    @abstractmethod
    def __init__(self, config: Dict):
        """
        Initialize the unlearning algorithm.

        Args:
            config (Dict): Configuration parameters for the algorithm.
        """
        pass

    @abstractmethod
    def _setup_components(self):
        """
        Set up the components of the unlearning algorithm, including the model, trainer, and sampler.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run the unlearning algorithm.
        """
        pass
