# mu_attack/core/base_task.py
from abc import ABC, abstractmethod

class Task(ABC):
    """
    A fully abstract base class defining the interface for any task.
    Each derived class must implement the following abstract methods:
        - load_models
        - get_loss
        - sample
        - evaluate
    """

    def __init__(
        self,
        *args, **kwargs
    ):
        pass


    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Must implement the final evaluation step (e.g., produce images, 
        measure metrics, or return success indicators).
        """
        pass
