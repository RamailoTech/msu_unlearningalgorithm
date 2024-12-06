import os
from typing import Any, Tuple, List
from PIL import Image
import torch
from einops import rearrange

from .base_dataset import BaseDataset

class GenericImageDataset(BaseDataset):
    """
    Generic dataset for image-based tasks.
    """

    def __init__(self, data_path: str, prompt_path: str, transform: Any = None):
        super().__init__(data_path, prompt_path, transform)
        self.image_paths = self._read_text_lines(data_path)
        self.prompts = self._read_text_lines(prompt_path)
        self.transform = transform

        assert len(self.image_paths) == len(self.prompts), "Mismatch between images and prompts."

    def _read_text_lines(self, path: str) -> List[str]:
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Normalize and rearrange image tensor
        image = rearrange(2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w")

        return image, prompt
