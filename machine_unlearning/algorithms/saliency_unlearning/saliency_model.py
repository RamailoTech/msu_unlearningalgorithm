# saliency_model.py

import torch
from torch import nn
from typing import Any

class SaliencyModel(BaseModel):
    def __init__(self):
        super(SaliencyModel, self).__init__()
        self.model = None
        self.device = None

    def load_model(self, config_path: str, ckpt_path: str, device: str):
        # Assuming a function `setup_model` is available to load the model
        self.model = setup_model(config_path, ckpt_path, device)
        self.device = device
        self.model.to(device)

    def save_model(self, output_path: str):
        torch.save({"state_dict": self.model.state_dict()}, output_path)

    def forward(self, input_data: Any) -> Any:
        return self.model(input_data)

    def get_parameters(self):
        # Returns model parameters for optimization
        return self.model.parameters()
