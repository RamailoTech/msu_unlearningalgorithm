# saliency_data_handler.py

import os
from typing import Any, Tuple, List
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class SaliencyDataHandler(BaseDataHandler):
    def __init__(self, forget_data_dir: str, remain_data_dir: str, image_size: int):
        self.forget_data_dir = forget_data_dir
        self.remain_data_dir = remain_data_dir
        self.image_size = image_size

    def load_data(self, data_path: str) -> List[Any]:
        images_path = os.path.join(data_path, 'images.txt')
        prompts_path = os.path.join(data_path, 'prompts.txt')
        with open(images_path, 'r') as f:
            image_paths = f.read().splitlines()
        with open(prompts_path, 'r') as f:
            prompts = f.read().splitlines()
        return list(zip(image_paths, prompts))

    def preprocess_data(self, data: List[Any]) -> List[Any]:
        preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        preprocessed_data = []
        for image_path, prompt in data:
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)
            preprocessed_data.append((image, prompt))
        return preprocessed_data

    def get_data_loaders(self, batch_size: int) -> Tuple[Any, Any]:
        class ImagePromptDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        forget_data = self.load_data(self.forget_data_dir)
        forget_data = self.preprocess_data(forget_data)
        forget_dataset = ImagePromptDataset(forget_data)
        forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)

        remain_data = self.load_data(self.remain_data_dir)
        remain_data = self.preprocess_data(remain_data)
        remain_dataset = ImagePromptDataset(remain_data)
        remain_loader = DataLoader(remain_dataset, batch_size=batch_size, shuffle=True)

        return forget_loader, remain_loader
