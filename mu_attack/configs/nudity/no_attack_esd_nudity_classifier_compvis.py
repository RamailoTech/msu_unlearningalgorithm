# mu_attack/configs/nudity/no_attack_esd_nudity_classifier_compvis.py

import os
from mu_attack.core.base_config import BaseConfig, OverallConfig, TaskConfigCompvis, NoAttackAttackerConfig, LoggerConfig


class NoAttackESDNudityClassifierConfigCompvis(BaseConfig):
    def __init__(self,
                 overall=None,
                 task=None,
                 attacker=None,
                 logger=None):
        super().__init__()

        # Overall Configuration
        self.overall = OverallConfig(
            task="classifier",
            attacker="no_attack",
            logger="json",
            resume=None,
            **(overall or {})
        )

        # Task Configuration
        self.task = TaskConfigCompvis(
            concept="nudity",
            compvis_ckpt_path="outputs/esd/esd_Abstractionism_model.pth",
            compvis_config_path="mu/algorithms/esd/configs/model_config.yaml",
            cache_path=".cache",
            dataset_path="outputs/dataset/i2p_nude",
            criterion="l1",
            classifier_dir=None,
            sampling_step_num=1,
            sld="weak",
            sld_concept="nudity",
            negative_prompt="sth",
            backend="compvis",
            **(task or {})
        )

        # Attacker Configuration
        self.attacker = NoAttackAttackerConfig(
            insertion_location="prefix_k",
            k=5,
            iteration=40,
            seed_iteration=1,
            attack_idx=1,
            eval_seed=0,
            universal=False,
            sequential=True,
            no_attack={
                "dataset_path": "outputs/dataset/i2p_nude"
            },
            **(attacker or {})
        )

        # Logger Configuration
        self.logger = LoggerConfig(
            json_config={
                "root": "results/no_attack_esd_nudity_esd_compvis",
                "name": "NoAttackEsdNudity"
            },
            **(logger or {})
        )

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if self.task:
            if not os.path.exists(self.task.compvis_ckpt_path):
                raise FileNotFoundError(f"Checkpoint path {self.task.compvis_ckpt_path} does not exist.")
            if not os.path.exists(self.task.compvis_config_path):
                raise FileNotFoundError(f"Config path {self.task.compvis_config_path} does not exist.")
            if not os.path.exists(self.task.dataset_path):
                raise FileNotFoundError(f"Dataset path {self.task.dataset_path} does not exist.")
        if self.logger and self.logger.json:
            if not os.path.exists(self.logger.json.root):
                raise FileNotFoundError(f"Logger root directory {self.logger.json.root} does not exist.")

    def to_dict(self):
        """
        Convert the entire configuration object to a dictionary.
        """

        # Handle attacker configuration explicitly
        attacker_dict = {
            "insertion_location": self.attacker.insertion_location,
            "k": self.attacker.k,
            "seed_iteration": self.attacker.seed_iteration,
            "sequential": self.attacker.sequential,
            "iteration": self.attacker.iteration,
            "attack_idx": self.attacker.attack_idx,
            "eval_seed": self.attacker.eval_seed,
            "universal": self.attacker.universal,
            "no_attack": vars(self.attacker.no_attack) if self.attacker.no_attack else None,
        }
        logger_dict = {
            "json": vars(self.logger.json) if self.logger.json else None
        }


        return {
            "overall": vars(self.overall),
            "task": vars(self.task),
            "attacker": attacker_dict,
            "logger": logger_dict,
        }



# Default instance of the configuration
no_attack_esd_nudity_classifier_compvis = NoAttackESDNudityClassifierConfigCompvis()
