# mu_attack/configs/nudity/hard_prompt_esd_nudity_P4D_compvis.py

import os
from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDNudityP4DConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="",
        backend="compvis"
    )

    attacker: AttackerConfig = AttackerConfig(
        lr=0.01,
        weight_decay=0.1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_nudity_P4D_scissorhands", "name": "P4d"}
    )
    
    def validate_config(self):
        """
        Validates the configuration, ensuring all required file paths exist.
        """
        if not os.path.exists(self.task["compvis_ckpt_path"]):
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.task.compvis_ckpt_path}")
        if not os.path.exists(self.task["compvis_config_path"]):
            raise FileNotFoundError(f"Config path does not exist: {self.task.compvis_config_path}")
        if not os.path.exists(self.task['dataset_path']):
            raise FileNotFoundError(f"Dataset path does not exist: {self.task.dataset_path}")


hard_prompt_esd_nudity_P4D_compvis_config = HardPromptESDNudityP4DConfigCompvis()
