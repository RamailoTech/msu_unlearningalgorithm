# mu_attack/configs/nudity/hard_prompt_esd_nudity_P4D_diffuser.py


import os
from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDNudityP4DConfigDiffusers(BaseConfig):
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
        backend="diffusers"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        lr=0.01,
        weight_decay=0.1
    )

    logger: LoggerConfig = LoggerConfig(
        json={
            "root": "results/hard_prompt_esd_nudity_P4D_scissorhands",
            "name": "P4d"
            }
    )

hard_prompt_esd_nudity_P4D_diffusers_config = HardPromptESDNudityP4DConfigDiffusers()
