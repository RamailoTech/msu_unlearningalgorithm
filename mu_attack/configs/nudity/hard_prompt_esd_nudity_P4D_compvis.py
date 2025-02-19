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
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False,
        converted_model_folder_path = "outputs"

    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        lr=0.01,
        weight_decay=0.1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_nudity_P4D_scissorhands", "name": "P4d"}
    )

hard_prompt_esd_nudity_P4D_compvis_config = HardPromptESDNudityP4DConfigCompvis()
