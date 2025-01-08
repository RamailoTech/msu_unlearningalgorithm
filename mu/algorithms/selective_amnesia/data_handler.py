#mu/algorithms/selective_amnesia/data_handler.py

from torch.utils.data import DataLoader
from functools import partial
import pytorch_lightning as pl
import os 
import pandas as pd 
import logging
from typing import Any


from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.data.base import Txt2ImgIterableBaseDataset
from stable_diffusion.scripts.txt2img_make_n_samples import main as f_txt_2_img_n_samples
from mu.algorithms.selective_amnesia.utils import worker_init_fn
from mu.algorithms.selective_amnesia.datasets import WrappedDataset

from mu.datasets.constants import * 

class SelectiveAmnesiaDataHandler(pl.LightningDataModule):
    """
    Concrete data handler for the Selective Amnesia algorithm.
    """
    def __init__(self, train_batch_size, val_batch_size, raw_dataset_dir, processed_dataset_dir,template,template_name, dataset_type,use_sample = False, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.raw_dataset_dir = raw_dataset_dir
        self.processed_dataset_dir = processed_dataset_dir
        self.template = template
        self.template_name = template_name
        self.dataset_type = dataset_type
        self.use_sample = use_sample

        self.logger = logging.getLogger(__name__)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else train_batch_size * 2
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        self.use_worker_init_fn = use_worker_init_fn
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
    def update_config_based_on_template(raw_dataset_dir, processed_dataset_dir, config, template, template_name, dataset_type, use_sample):
        if dataset_type == 'unlearncanvas':
            if template == 'object' : 
                raise ValueError("Only style template is supported for unlearncanvas dataset")
            
            if use_sample :
                assert template_name in uc_sample_theme_available, f"Invalid template name: {template_name}"
            else:
                assert template_name in uc_theme_available, f"Invalid template name: {template_name}"

            config.data.params.train.forget_prompt = f"An image in {template_name} style"

            config.data.params.validation.params.captions = [
                f"A {class_} image in {template_name} style"
                for class_ in (
                    uc_sample_class_available if use_sample else uc_class_available
                )
            ]       
            
        elif dataset_type == 'i2p': 

            prompts_file = os.path.join(raw_dataset_dir, 'prompts', 'i2p.csv')

            if not os.path.exists(prompts_file):
                raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

            # Read the CSV file
            data = pd.read_csv(prompts_file)

            categories = data['categories'].unique()

            assert template_name in categories, f"Invalid template name: {template_name}"
            config.data.params.train.params.forget_prompt = f"An image in {template_name} style"
            config.data.params.validation.params.captions = [
                f"A {class_} image in {template_name} style"
                for class_ in categories
            ]
        n_samples = 10 if use_sample else 1000
        model_config_path = config.model_config_path
        ckpt_path = config.ckpt_path
        SelectiveAmnesiaDataHandler.generate_dataset(outdir=f"{processed_dataset_dir}/{template_name}",prompt=config.data.params.train.forget_prompt,n_samples=n_samples,model_config_path=model_config_path, ckpt_path=ckpt_path)

        os.makedirs(f"{processed_dataset_dir}/replay_dataset", exist_ok=True)
        SelectiveAmnesiaDataHandler.generate_dataset(outdir=f"{processed_dataset_dir}/replay_dataset",from_file= config.replay_prompt_path,model_config_path=model_config_path, ckpt_path=ckpt_path)
        config.data.params.train.params.forget_dataset_path = f"{processed_dataset_dir}/{template_name}"
        config.data.params.train.params.replay_dataset_path = f"{processed_dataset_dir}/replay_dataset"
        config.data.params.train.params.replay_prompt_path = config.replay_prompt_path

        return config

    @staticmethod
    def generate_dataset(outdir, model_config_path, ckpt_path, from_file=None,prompt=None, n_samples=10):
        """
        Generate dataset based on the dataset type
        """
                
        try:
            if from_file: 
                f_txt_2_img_n_samples(outdir=outdir,config=model_config_path,ckpt=ckpt_path,from_file=from_file)
            else:
                f_txt_2_img_n_samples(outdir=outdir, prompt=prompt, n_samples= n_samples, config=model_config_path,ckpt=ckpt_path)
        except Exception as e:
            logging.error(f"Failed to generate dataset: {e}")
            raise

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
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
        return DataLoader(self.datasets["train"], batch_size=self.train_batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.val_batch_size,
                          num_workers=self.num_val_workers,
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
        return DataLoader(self.datasets["predict"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

