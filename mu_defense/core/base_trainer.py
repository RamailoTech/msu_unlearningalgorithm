from abc import ABC, abstractmethod
import logging

class BaseTrainer(ABC):
    """
    BaseTrainerRunner is an abstract base class for high-level training orchestrators.
    It defines the interface and common properties for running a training process.
    """
    def __init__(self, config: dict):
        self.config = config
        self.devices = config.get("devices", ["cuda:0"])
        self.logger = logging.getLogger(__name__)

    def train(self):
        pass
    
    @abstractmethod
    def run(self):
        """
        Run the training loop. Must be implemented by subclasses.
        """
        pass
