import logging
from typing import Dict
from torch.utils.data import DataLoader
from core.base_data_handler import BaseDataHandler
from datasets.transforms import get_transform, INTERPOLATIONS
from algorithms.selective_amnesia.datasets.dataset import SelectiveAmnesiaDataset

logger = logging.getLogger(__name__)

class SelectiveAmnesiaDataHandler(BaseDataHandler):
    """
    Data handler for Selective Amnesia.
    Loads the surrogate dataset q(x|c_f) and returns DataLoaders.
    """

    def __init__(
        self,
        surrogate_data_dir: str,
        image_size: int = 512,
        interpolation: str = 'bicubic',
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Args:
            surrogate_data_dir (str): Directory containing the surrogate dataset q(x|c_f).
            image_size (int, optional): Size to which images are resized.
            interpolation (str, optional): Interpolation mode.
            batch_size (int, optional): Batch size for the DataLoader.
            num_workers (int, optional): Number of worker threads.
            pin_memory (bool, optional): Use pinned memory.
        """
        self.surrogate_data_dir = surrogate_data_dir
        self.image_size = image_size
        self.interpolation = interpolation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation}. "
                             f"Supported: {list(INTERPOLATIONS.keys())}")

        self.data_loaders = self.get_data_loaders()

    def get_data_loaders(self) -> Dict[str, DataLoader]:
        transform = get_transform(interpolation=INTERPOLATIONS[self.interpolation],
                                  size=self.image_size)

        dataset = SelectiveAmnesiaDataset(images_dir=self.surrogate_data_dir, transform=transform)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        # If you have separate validation or test sets, create them similarly.
        return {'train': train_loader}
