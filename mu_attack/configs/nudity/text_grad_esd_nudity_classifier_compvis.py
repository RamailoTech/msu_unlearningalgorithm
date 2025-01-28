# mu_attack/configs/nudity/text_grad_esd_nudity_classifier_compvis.py

import os
from mu_attack.core.base_config import BaseConfig, OverallConfig, TaskConfigCompvis, AttackerConfig, LoggerConfig


class TextGradESDNudityClassifierCompvis(BaseConfig):
    def __init__(self,
                 overall=None,
                 task=None,
                 attacker=None,
                 logger=None):
        super().__init__()

        self.overall = OverallConfig(
            task="classifier",
            attacker="text_grad",
            logger="json",
            resume=None,
            **(overall or {})
        )

        self.task = TaskConfigCompvis(
            concept="nudity",
            compvis_ckpt_path="outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
            compvis_config_path="mu/algorithms/scissorhands/configs/model_config.yaml",
            cache_path=".cache",
            dataset_path="outputs/dataset/i2p_nude",
            criterion="l2",
            sampling_step_num=1,
            sld="weak",
            sld_concept="nudity",
            negative_prompt="sth",
            backend="compvis",
            classifier_dir=None,
            **(task or {})
        )

        self.attacker = AttackerConfig(
            insertion_location="prefix_k",
            k=5,
            iteration=1,
            seed_iteration=1,
            attack_idx=1,
            eval_seed=0,
            universal=False,
            sequential=True,
            text_grad= {
            "lr": 0.01,
            "weight_decay": 0.1
            },
            **(attacker or {})
        )

        self.logger = LoggerConfig(
            json_config={
                "root": "results/seed_search_esd_nudity_scissorhands",
                "name": "TextGradNudity"
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


text_grad_esd_nudity_classifier_compvis = TextGradESDNudityClassifierCompvis()
