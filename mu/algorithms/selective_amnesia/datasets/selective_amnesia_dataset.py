import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Callable

class SelectiveAmnesiaDataset(Dataset):
    """
    Dataset for the Selective Amnesia forgetting task.
    This dataset can load images and prompts from a generated surrogate dataset q(x|c_f).
    """

    def __init__(self, images_dir: str, transform: Callable = None):
        """
        Args:
            images_dir (str): Directory containing the surrogate dataset images.
            transform (Callable, optional): Transformations to apply to images.
        """
        self.images_dir = images_dir
        self.transform = transform
        # Assume images_dir has been populated by a prior generation step.
        # If you need prompts, store them similarly or generate a prompts.txt.

        self.image_paths = sorted([os.path.join(self.images_dir, f) 
                                   for f in os.listdir(self.images_dir)
                                   if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # If prompts are required, load them similarly from a txt file or other source.
        # For now, we assume the surrogate dataset doesn't need prompts.
        return image
