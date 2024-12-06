from typing import Any, Tuple
import torch

from mu.datasets.generic_dataset import GenericImageDataset
from mu.datasets.utils import get_logger

class EraseDiffDataset(GenericImageDataset):
    """
    Dataset tailored for the EraseDiff algorithm.
    Inherits from GenericImageDataset and can include additional processing if needed.
    """
    def __init__(self, data_path: str, prompt_path: str, transform: Any = None, additional_param: Any = None):
        super().__init__(data_path, prompt_path, transform)
        self.logger = get_logger(self.__class__.__name__)
        self.additional_param = additional_param
        # Add any EraseDiff-specific initialization here
        self.logger.info(f"Initialized EraseDiffDataset with {len(self)} samples.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Optionally override to include EraseDiff-specific processing.
        """
        try:
            image, prompt = super().__getitem__(idx)
            # Implement any additional processing specific to EraseDiff
            return image, prompt
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise
