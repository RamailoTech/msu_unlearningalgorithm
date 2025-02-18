# mu_attack/core/base_evaluator.py

from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, config, **kwargs):
        """
        Initializes the evaluator with a configuration.
        :param config: A configuration object or dictionary.
        :param kwargs: Additional configuration overrides.
        """
        self.config = config
        self.results = {}
        self._parse_config(**kwargs)

    @abstractmethod
    def _parse_config(self, **kwargs):
        """Parse and validate configuration."""
        pass

    @abstractmethod
    def run_evaluation(self):
        """Run the evaluation process and set self.results accordingly."""
        pass

    def run(self):
        """Run the complete evaluation process."""
        pass