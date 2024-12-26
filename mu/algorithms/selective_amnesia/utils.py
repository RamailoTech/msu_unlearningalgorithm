import logging
import os

import torch

logger = logging.getLogger(__name__)

def load_fim(fim_path: str):
    """
    Load the precomputed Fisher Information Matrix (FIM).
    """
    if not os.path.exists(fim_path):
        raise FileNotFoundError(f"FIM file not found at {fim_path}")
    fim_dict = torch.load(fim_path, map_location='cpu')
    logger.info(f"Loaded FIM from {fim_path}")
    return fim_dict

def modify_weights(w, scale=1e-6, n=2):
    """
    Modify weights to accommodate changes in input channels (if needed).
    """
    extra_w = scale * torch.randn_like(w)
    new_w = w.clone()
    for _ in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w
