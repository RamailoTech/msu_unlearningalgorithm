# mu_attack/configs/nudity/random_esd_nudity_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig


class RandomESDNudityDiffuser(BaseConfig):
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
        backend="diffusers",
        target_ckpt="files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        attack_idx=1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/random_esd_nudity_scissorhands", "name": "Hard Prompt"}
    )

random_esd_nudity_diffuser_config = RandomESDNudityDiffuser()
