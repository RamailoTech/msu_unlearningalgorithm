# evaluation/core/base_evaluator.py

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):

    def __init__(self,config: Dict[str, Any], **kwargs):
        self.config = config

    @abstractmethod
    def load_and_prepare_data(self, *args, **kwargs):
        """
        Must implement the data loading and preparation process.
        """
        pass

    @abstractmethod
    def compute_score(self, *args, **kwargs):
        """
        Must implement the metric computation process.
        """
        pass

    @abstractmethod
    def save_results(self, *args, **kwargs):
        """
        Must implement the saving process.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the evaluation process."""
        pass