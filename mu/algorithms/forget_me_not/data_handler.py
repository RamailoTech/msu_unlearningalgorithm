# forget_me_not/data_handler.py

import os
from typing import Any, Dict

from algorithms.forget_me_not.datasets.forget_me_not_dataset import ForgetMeNotDataset
from core.base_data_handler import BaseDataHandler
from datasets.constants import *
from torch.utils.data import DataLoader


class ForgetMeNotDataHandler(BaseDataHandler):
    """
    Data Handler for the Forget Me Not algorithm.
    Extends the BaseDataHandler to manage data loading, preprocessing, and data loader creation.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.original_data_dir = self.config.get("original_data_dir", ".")
        self.new_data_dir = self.config.get("new_data_dir", ".")
        self.theme = self.config.get("theme", "")
        self.classes = self.config.get("classes", "")
        self.use_sample = self.config.get("use_sample", False)

    def setup(self, **kwargs):
        """
        Perform any setup required before loading/preprocessing data.
        """
        # If any special setup is needed before data loading, implement here.
        self.generate_dataset()

    def load_data(self, data_path: str) -> Any:
        """
        Load data from the specified path.

        Args:
            data_path (str): Path to the data.

        Returns:
            Any: Loaded data (e.g., dataset object).
        """
        # In this example, we'll rely on the dataset itself to handle loading logic.
        # If you need to load from disk or do something special before creating the dataset, implement here.
        return None

    def preprocess_data(self, data: Any) -> Any:
        """
        Preprocess the data (e.g., normalization, augmentation).

        Args:
            data (Any): Raw data to preprocess.

        Returns:
            Any: Preprocessed data.
        """
        # Preprocessing is handled within the ForgetMeNotDataset transformations.
        # If additional preprocessing is required outside the dataset, implement here.
        return data

    def get_data_loaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """
        Create and return data loaders for training (and optionally validation/test sets).

        Args:
            batch_size (int): Batch size for data loaders.

        Returns:
            Dict[str, DataLoader]: A dictionary of data loaders.
        """
        # Instantiate the ForgetMeNotDataset with parameters from config.
        dataset = ForgetMeNotDataset(
            instance_data_root=self.config.get("instance_data_dir", ""),
            tokenizer=self.config.get(
                "tokenizer"
            ),  # Ensure tokenizer is passed in config
            token_map=self.config.get("token_map"),
            use_template=self.config.get("use_template"),
            class_data_root=self.config.get("class_data_root"),
            class_prompt=self.config.get("class_prompt"),
            size=self.config.get("size", 512),
            h_flip=self.config.get("h_flip", True),
            color_jitter=self.config.get("color_jitter", False),
            resize=self.config.get("resize", True),
            use_face_segmentation_condition=self.config.get(
                "use_face_segmentation_condition", False
            ),
            blur_amount=self.config.get("blur_amount", 70),
        )

        # Create a DataLoader for the dataset
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
        )

        return {"train": train_loader}

    def generate_dataset(self):
        """
        Generate datasets by organizing images into themes and classes.
        This method encapsulates the dataset generation logic.
        """
        dataset_dir = self.original_data_dir

        # Ensure the base data directory exists
        os.makedirs(self.new_data_dir, exist_ok=True)

        # Generate datasets for themes
        for theme in uc_sample_theme_available:
            theme_dir = os.path.join(self.new_data_dir, theme)
            os.makedirs(theme_dir, exist_ok=True)

            for i, object_class in enumerate(uc_sample_class_available):
                source_file = os.path.join(dataset_dir, theme, object_class, "1.jpg")
                target_file = os.path.join(theme_dir, f"{i}.jpg")

                if os.path.exists(source_file):
                    os.system(f"cp {source_file} {target_file}")
                else:
                    self.logger.warning(f"Source file not found: {source_file}")

        # Generate datasets for object classes
        for object_class in uc_sample_class_available:
            for i, theme in enumerate(uc_sample_theme_available):
                source_file = os.path.join(dataset_dir, theme, object_class, "1.jpg")
                target_file = os.path.join(self.new_data_dir, theme, f"{i}.jpg")

                if os.path.exists(source_file):
                    os.system(f"cp {source_file} {target_file}")
                else:
                    self.logger.warning(f"Source file not found: {source_file}")

        self.logger.info("Dataset generation completed.")
