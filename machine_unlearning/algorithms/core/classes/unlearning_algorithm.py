from abc import ABC, abstractmethod

class UnlearningAlgorithm(ABC):
    """
    Abstract base class for the overall unlearning algorithm, combining the model, trainer, and sampler.
    All algorithms must inherit from this class and implement its methods.
    """

    @abstractmethod
    def __init__(self, config):
        """
        Initialize the unlearning algorithm.
        Args:
            config (dict): Configuration parameters for the algorithm.
        """
        pass

    @abstractmethod
    def run(self, config):
        """
        Run the unlearning algorithm with the specified configuration.
        Args:
            config (dict): Configuration parameters for running the algorithm.
        """
        pass
