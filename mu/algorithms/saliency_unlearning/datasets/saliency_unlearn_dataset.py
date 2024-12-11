# algorithms/saliency_unlearning/datasets/saliency_unlearn_dataset.py

import os
import torch 
from typing import Any, Tuple, Dict
from torch.utils.data import DataLoader
from datasets.unlearn_canvas_dataset import UnlearnCanvasDataset
from datasets.transforms import INTERPOLATIONS, get_transform

class SaliencyUnlearnDataset(UnlearnCanvasDataset):
    """
    Dataset class for the SaliencyUnlearn algorithm.
    Extends UnlearnCanvasDataset to handle specific requirements.
    Manages both 'forget' and 'remain' datasets, with support for masks.
    """

    def __init__(
        self,
        forget_data_dir: str,
        remain_data_dir: str,
        mask_path: str,
        selected_theme: str,
        selected_class: str,
        use_sample: bool = False,
        image_size: int = 512,
        interpolation: str = 'bicubic',
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize the SaliencyUnlearnDataset.

        Args:
            forget_data_dir (str): Directory containing forget dataset.
            remain_data_dir (str): Directory containing remain dataset.
            mask_path (str): Path to the mask file.
            selected_theme (str): Theme to filter images.
            selected_class (str): Class to filter images.
            use_sample (bool, optional): Whether to use sample constants. Defaults to False.
            image_size (int, optional): Size to resize images. Defaults to 512.
            interpolation (str, optional): Interpolation mode for resizing. Defaults to 'bicubic'.
            batch_size (int, optional): Batch size for data loaders. Defaults to 4.
            num_workers (int, optional): Number of worker threads for data loading. Defaults to 4.
            pin_memory (bool, optional): Whether to pin memory in DataLoader. Defaults to True.
        """
        # Initialize transformations
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"Unsupported interpolation mode: {interpolation}. Supported modes: {list(INTERPOLATIONS.keys())}")

        interpolation_mode = INTERPOLATIONS[interpolation]
        transform = get_transform(interpolation=interpolation_mode, size=image_size)

        # Initialize forget dataset
        self.forget_dataset = UnlearnCanvasDataset(
            data_dir=forget_data_dir,
            selected_theme=selected_theme,
            selected_class=selected_class,
            use_sample=use_sample,
            transform=transform
        )

        # Initialize remain dataset
        self.remain_dataset = UnlearnCanvasDataset(
            data_dir=remain_data_dir,
            selected_theme=selected_theme,
            selected_class=selected_class,
            use_sample=use_sample,
            transform=transform
        )

        # Load mask
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask file not found at {mask_path}")
        self.mask = torch.load(mask_path)

        # Initialize DataLoaders
        self.forget_loader = DataLoader(
            self.forget_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.remain_loader = DataLoader(
            self.remain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Retrieve the forget and remain data loaders.

        Returns:
            Dict[str, DataLoader]: Dictionary containing 'forget' and 'remain' DataLoaders.
        """
        return {
            'forget': self.forget_loader,
            'remain': self.remain_loader
        }

    def __len__(self) -> int:
        """
        Returns the length based on the forget dataset.

        Returns:
            int: Number of samples in the forget dataset.
        """
        return len(self.forget_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Retrieve a sample from the forget dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, str]: A tuple containing the data sample and its corresponding prompt.
        """
        return self.forget_dataset[idx]

    def get_mask(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve the mask associated with the forget dataset.

        Returns:
            Dict[str, torch.Tensor]: Mask dictionary.
        """
        return self.mask
