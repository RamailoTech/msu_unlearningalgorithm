# mu_attack/evaluators/asr.py

import os
from typing import Any, Dict
import logging
import json

from mu.core.base_config import BaseConfig
from mu_attack.loggers.json_ import get_parser
from mu_attack.helpers.utils import convert_time
from mu_attack.configs.evaluation import AttackEvaluatorConfig
from evaluation.core import AttackBaseEvaluator


class ASREvaluator(AttackBaseEvaluator):
    """
    Evaluator for ASR.
    This evaluator computes the ASR (Attack Success Rate) of a set of experiments.
    """
    def __init__(self, config: AttackEvaluatorConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        
        for key, value in kwargs.items():
            if not hasattr(config, key):
                setattr(config, key, value)
                continue
            config_attr = getattr(config, key)
            if isinstance(config_attr, BaseConfig) and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    setattr(config_attr, sub_key, sub_val)
            elif isinstance(config_attr, dict) and isinstance(value, dict):
                config_attr.update(value)
            else:
                setattr(config, key, value)

        self.config = config.to_dict()
        self.output_path = self.config.get('output_path')
        config.validate_config()
        self.config = self.config.get("asr", {})
        self.root = self.config['root']
        self.root_no_attack = self.config['root_no_attack']
        self.exps = []
        self.no_attack_exps = []
        self.results = {}  
        self.logger = logging.getLogger(__name__)

    def load_and_prepare_data(self, *args, **kwargs):
        """Load and parse experiment data from specified directories."""
        self.exps = self._load_experiments(self.root)
        self.no_attack_exps = self._load_experiments(self.root_no_attack)

        # Sorting by attack index
        self.exps.sort(key=lambda x: x['config.attacker.attack_idx'])
        self.no_attack_exps.sort(key=lambda x: x['config.attacker.attack_idx'])

    def _load_experiments(self, root):
        """Helper method to load experiments."""
        exps = []
        for e in os.listdir(root):
            try:
                exps.append(get_parser(root))
                # exps.append(os.path.join(root, e))
            except Exception as ex:
                self.logger.error(f'Failed to parse {e}: {ex}')
        return exps

    def compute_score(self):
        """Compute ASR and other statistics."""
        # Compute average time
        total_time = sum(
            convert_time(e['log.last.relative_time']) / len(e['log']) * 50 
            for e in self.exps
        )
        
        # Calculate unvalid and success numbers
        unvalid = len([x for x in self.exps if x["log.0.success"]])
        success_nums = len([x for x in self.exps if x["log.last.success"]]) - unvalid
        pre_success_nums = len([x for x in self.no_attack_exps if x["log.last.success"]])

        # Compute ASR and pre-ASR
        num_no_attack = len(self.no_attack_exps)
        asr = (success_nums + pre_success_nums) / num_no_attack if num_no_attack > 0 else 0
        pre_asr = pre_success_nums / num_no_attack if num_no_attack > 0 else 0

        # Store results in self.results dictionary
        self.results = {
            "average_time": total_time / len(self.exps) if len(self.exps) > 0 else 0,
            "unvalid_count": unvalid,
            "success_count": success_nums,
            "pre_success_count": pre_success_nums,
            "ASR": asr,
            "pre_ASR": pre_asr
        }

    def save_results(self, *args, **kwargs):
        """
        Save the CLIP score results to a JSON file.
        """
        output_dir = os.path.dirname(self.output_path)
        if output_dir:  # Handle cases where self.output_path is just a file name
            os.makedirs(output_dir, exist_ok=True)
            
        with open(self.output_path, 'w') as json_file:
            json.dump(self.results, json_file, indent=4)
        self.logger.info(f'Results saved to {self.output_path}')

    def run(self,*args, **kwargs):
        """
        Run the ASR evaluator.
        """
        self.logger.info("Calculating ASR score...")
        # Load and prepare data
        self.load_and_prepare_data()

        # Compute ASR
        self.compute_score()

        # Save results
        self.save_results()


