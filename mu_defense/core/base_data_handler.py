# mu_defense/core/base_dataset_handler.py

from abc import ABC, abstractmethod

class BaseDatasetHandler(ABC):
    """
    BaseDatasetHandler provides a blueprint for handling dataset-related tasks,
    including prompt cleaning and creation of a retaining dataset.
    """
    def __init__(self, prompt: str, seperator: str = None, dataset_retain=None):
        self.prompt = prompt
        self.seperator = seperator
        self.dataset_retain = dataset_retain
        self.words = []
        self.word_print = ""

    @abstractmethod
    def setup_prompt(self):
        """
        Set up and return the cleaned prompt and the printable version.
        """
        pass

    @abstractmethod
    def setup_dataset(self):
        """
        Create and return the retaining dataset.
        """
        pass