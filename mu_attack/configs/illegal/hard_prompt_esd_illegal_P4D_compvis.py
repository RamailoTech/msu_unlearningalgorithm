# mu_attack/configs/illegal/hard_prompt_esd_illegal_P4D_compvis.py

import os
from mu_attack.core.base_config import BaseConfig, OverallConfig, TaskConfigCompvis, AttackerConfig, LoggerConfig


class HardPromptIllegalConfigCompvis(BaseConfig):
    def __init__(self,
                 overall=None,
                 task=None,
                 attacker=None,
                 logger=None):
        super().__init__()

        self.overall = OverallConfig(
            task="P4D",
            attacker="hard_prompt",
            logger="json",
            resume=None,
            **(overall or {})
        )

        self.task = TaskConfigCompvis(
            concept="harm",
            compvis_ckpt_path="outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
            compvis_config_path="mu/algorithms/scissorhands/configs/model_config.yaml",
            cache_path=".cache",
            dataset_path="outputs/dataset/illegal",
            criterion="l2",
            classifier_dir=None,
            backend="compvis",
            **(task or {})
        )

        self.attacker = AttackerConfig(
            insertion_location="prefix_k",
            k=5,
            iteration=40,
            seed_iteration=1,
            attack_idx=0,
            eval_seed=0,
            universal=False,
            sequential=True,
            hard_prompt= {
                "lr": 0.01,
                "weight_decay": 0.1
                }
            **(attacker or {})
        )

        self.logger = LoggerConfig(
            json_config={
                "root": "files/results/hard_prompt_esd_illegal_P4D_scissorhands"
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
        return {
            "overall": {
                "task": self.overall.task,
                "attacker": self.overall.attacker,
                "logger": self.overall.logger,
                "resume": self.overall.resume,
            },
            "task": vars(self.task),
            "attacker": vars(self.attacker),
            "logger": {
                "json": vars(self.logger.json)
            },
        }



hard_prompt_esd_illegal_P4D_compvis_config = HardPromptIllegalConfigCompvis()
