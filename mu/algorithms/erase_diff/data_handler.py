# erasediff_data_handler.py

from algorithms.core.base_data_handler import BaseDataHandler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

class EraseDiffDataHandler(BaseDataHandler):
    def __init__(self, forget_data_dir: str, remain_data_dir: str, batch_size: int, image_size: int):
        self.forget_data_dir = forget_data_dir
        self.remain_data_dir = remain_data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.forget_dl = None
        self.remain_dl = None

    def load_data(self, data_path: str):
        dataset = datasets.ImageFolder(root=data_path, transform=self.transform)
        return dataset

    def get_data_loaders(self):
        forget_dataset = self.load_data(self.forget_data_dir)
        remain_dataset = self.load_data(self.remain_data_dir)
        self.forget_dl = DataLoader(forget_dataset, batch_size=self.batch_size, shuffle=True)
        self.remain_dl = DataLoader(remain_dataset, batch_size=self.batch_size, shuffle=True)
        return self.forget_dl, self.remain_dl
