# mu_attack/configs/nudity/random_esd_nudity_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig


class RandomESDNudityCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="random",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        attack_idx=1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/random_esd_nudity_scissorhands", "name": "Hard Prompt"}
    )

random_esd_nudity_compvis_config = RandomESDNudityCompvis()
