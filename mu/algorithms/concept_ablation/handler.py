import logging
import os
from typing import Dict

from algorithms.concept_ablation.datasets.dataset import ConceptAblationDataset
from core.base_data_handler import BaseDataHandler
from torch.utils.data import DataLoader

from mu.datasets.utils import INTERPOLATIONS, get_transform


class ConceptAblationDataHandler(BaseDataHandler):
    """
    Data handler for the Concept Ablation algorithm.
    Manages data generation, filtering, and DataLoader creation.
    """

    def __init__(
        self,
        concept_type: str,
        prompts_path: str,
        output_dir: str,
        base_config: str,
        resume_ckpt: str,
        delta_ckpt: str = None,
        caption_target: str = None,
        train_size: int = 1000,
        n_samples: int = 10,
        image_size: int = 512,
        interpolation: str = 'bicubic',
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_regularization: bool = False
    ):
        """
        Initialize the ConceptAblationDataHandler.

        Args:
            concept_type (str): Type of concept being removed ('style', 'object', or 'memorization').
            prompts_path (str): Path to text file containing initial prompts.
            output_dir (str): Directory to store generated images and associated metadata.
            base_config (str): Path to the Stable Diffusion base config file.
            resume_ckpt (str): Path to the Stable Diffusion checkpoint to resume from.
            delta_ckpt (str, optional): Path to a delta checkpoint for additional fine-tuning. Defaults to None.
            caption_target (str, optional): Target concept/style to remove. Used for filtering. Defaults to None.
            train_size (int, optional): Number of training images to generate. Defaults to 1000.
            n_samples (int, optional): Number of images per generation step. Defaults to 10.
            image_size (int, optional): Size to which images are resized. Defaults to 512.
            interpolation (str, optional): Interpolation mode for resizing. Defaults to 'bicubic'.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
            num_workers (int, optional): Number of DataLoader workers. Defaults to 4.
            pin_memory (bool, optional): Whether to use pinned memory in DataLoader. Defaults to True.
            use_regularization (bool, optional): Whether to add a regularization dataset. Defaults to False.
        """
        self.concept_type = concept_type
        self.prompts_path = prompts_path
        self.output_dir = output_dir
        self.base_config = base_config
        self.resume_ckpt = resume_ckpt
        self.delta_ckpt = delta_ckpt
        self.caption_target = caption_target
        self.train_size = train_size
        self.n_samples = n_samples
        self.image_size = image_size
        self.interpolation = interpolation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_regularization = use_regularization

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ConceptAblationDataHandler...")

        self.data_loaders = self.get_data_loaders()

    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Create and return the DataLoaders for the concept ablation dataset.

        Returns:
            Dict[str, DataLoader]: Dictionary containing at least a 'train' DataLoader.
        """
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation}. "
                             f"Supported modes: {list(INTERPOLATIONS.keys())}")

        transform = get_transform(interpolation=INTERPOLATIONS[self.interpolation], size=self.image_size)

        # Instantiate the dataset. This will handle generation and filtering internally.
        dataset = ConceptAblationDataset(
            concept_type=self.concept_type,
            prompts_path=self.prompts_path,
            output_dir=self.output_dir,
            base_config=self.base_config,
            resume_ckpt=self.resume_ckpt,
            delta_ckpt=self.delta_ckpt,
            caption_target=self.caption_target,
            train_size=self.train_size,
            n_samples=self.n_samples,
            image_size=self.image_size,
            transform=transform,
            use_regularization=self.use_regularization
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        return {'train': train_loader}
