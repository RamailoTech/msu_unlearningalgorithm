# mu_attack/configs/nudity/text_grad_esd_nudity_classifier_diffuser.py

import os
from mu_attack.core.base_config import BaseConfig, OverallConfig, TaskConfigDiffuser, AttackerConfig, LoggerConfig


class TextGradESDNudityClassifierDiffuser(BaseConfig):
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

        # Task configuration
        self.task = TaskConfigDiffuser(
            concept="nudity",
            diffusers_model_name_or_path="outputs/forget_me_not/finetuned_models/Abstractionism",
            target_ckpt="files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
            cache_path=".cache",
            dataset_path="outputs/dataset/i2p_nude",
            criterion="l2",
            sampling_step_num=1,
            classifier_dir = None,
            sld="weak",
            sld_concept="nudity",
            negative_prompt="sth",
            backend="diffusers",
            **(task or {})
        )

        # Attacker configuration
        self.attacker = AttackerConfig(
            insertion_location="prefix_k",
            k=5,
            iteration=1,
            seed_iteration=1,
            attack_idx=0,
            eval_seed=0,
            universal=False,
            sequential=True,
            text_grad= {
            "lr": 0.01,
            "weight_decay": 0.1
            },
            **(attacker or {})
        )

        # Logger configuration
        self.logger = LoggerConfig(
            json_config={
                "root": "results/random_esd_nudity_abstractionism",
                "name": "TextGradNudity"
            },
            **(logger or {})
        )

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if self.task:
            if not os.path.exists(self.task.diffusers_model_name_or_path):
                raise FileNotFoundError(f"Diffusers model path {self.task.diffusers_model_name_or_path} does not exist.")
            if not os.path.exists(self.task.target_ckpt):
                raise FileNotFoundError(f"Target checkpoint {self.task.target_ckpt} does not exist.")
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

# Default instance of the configuration
text_grad_esd_nudity_classifier_diffuser = TextGradESDNudityClassifierDiffuser()
