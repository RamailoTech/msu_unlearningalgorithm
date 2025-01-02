import torch
from stable_diffusion.ldm.data.base import Txt2ImgIterableBaseDataset
import numpy as np

def modify_weights(w, scale=1e-6, n=2):
    """
    Modify weights to accommodate changes in input channels (if needed).
    """
    extra_w = scale * torch.randn_like(w)
    new_w = w.clone()
    for _ in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
