from abc import ABC, abstractmethod


class BaseConfig(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def validate_config(self):
        pass
