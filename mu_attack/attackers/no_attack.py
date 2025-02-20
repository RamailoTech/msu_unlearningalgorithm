#mu_attack/attackers/no_attack.py

import logging 

import torch
import torch.nn.functional as F
import numpy as np

from mu_attack.core import Attacker
from mu_attack.helpers import get_dataset

class NoAttacker(Attacker):
    def __init__(
                self,
                dataset_path,
                **kwargs
                ):
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)
        super().__init__(**kwargs)
        
    def run(self, task, logger):
        task.dataset = get_dataset(self.dataset_path)
        image, prompt, seed, guidance = task.dataset[self.attack_idx]
        if seed is None:
            seed = self.eval_seed
            
        task.pipe.tokenizer.pad_token = task.pipe.tokenizer.eos_token
        
        viusalize_prompt_id = task.pipe.str2id(prompt)

        ### Visualization for the original prompt:
        results = task.evaluate(viusalize_prompt_id,prompt,seed=seed,guidance_scale=guidance)
        results['prompt'] = prompt
        logger.save_img('orig', results.pop('image'))
        logger.log(results)

def get(**kwargs):
    return NoAttacker(**kwargs)