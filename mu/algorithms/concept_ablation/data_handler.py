#mu/algorithms/concept_ablation/data_handler.py
import os
import logging
from torch.utils.data import DataLoader
from functools import partial
import pytorch_lightning as pl
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np

from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.data.base import Txt2ImgIterableBaseDataset

from mu.helpers import safe_dir
from mu.algorithms.concept_ablation.utils import worker_init_fn
from mu.algorithms.concept_ablation.datasets.concept_ablation_dataset import WrappedDataset, ConcatDataset
from mu.algorithms.concept_ablation.utils import distributed_sample_images

class ConceptAblationDataHandler(pl.LightningDataModule):
    '''
        """
    Concrete data handler for the Concept Ablation algorithm.
    """'''
    def __init__(self, batch_size, train=None, train2=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.logger = logging.getLogger(__name__)

        if train2 is not None and train2['params']['caption'] != '':
            self.dataset_configs["train2"] = train2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    @staticmethod
    def preprocess(opt_config, model_config_path,outdir, ranks): 
        '''
        Preprocess data for the model.'''
        # mp.set_start_method('spawn')
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

        with open(opt_config.get("prompts"), "r") as f:
            data = f.read().splitlines()
            assert opt_config.get("train_size") % len(data) == 0
            n_repeat = opt_config.get("train_size") // len(data)
            data = np.array([n_repeat * [prompt] for prompt in data]
                            ).reshape(-1, opt_config.get("n_samples")).tolist()
        # check integrity
        sample_path = safe_dir(outdir / 'samples')
        if not sample_path.exists() or not len(list(sample_path.glob('*'))) == opt_config.get("train_size"):
            distributed_sample_images(
                data, ranks, model_config_path, opt_config.get("ckpt_path"),
                None, str(outdir), 200,10
            )


    def prepare_data(self):
        '''
        Prepare data for the model.'''
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        '''
        Setup data for the model.'''
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if "train2" in self.dataset_configs and self.dataset_configs["train2"]['params']["caption"] != '':
            train_set = self.datasets["train"]
            train2_set = self.datasets["train2"]
            concat_dataset = ConcatDataset(train_set, train2_set)
            return DataLoader(concat_dataset, batch_size=self.batch_size // 2,
                              num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                              worker_init_fn=init_fn)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                              worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)
