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
        self.config = config

    def _parse_config(self):
        """
        Parse the configuration parameters for the algorithm.
        """
        # Parse devices
        devices = [
            f"cuda:{int(d.strip())}" for d in self.config.get("devices", "0").split(",")
        ]
        self.config["devices"] = devices

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
