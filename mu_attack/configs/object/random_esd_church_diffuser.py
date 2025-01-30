# mu_attack/configs/object/random_esd_church_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class RandomESDChurchConfigDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="random",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        concept = "church",
        backend="diffusers",
        target_ckpt = "files/pretrained/SD-1-4/ESD_ckpt/Church-ESDu1-UNET-SD.pt",
    )

    attacker: AttackerConfig = AttackerConfig(
        k = 3,
        attack_idx = 1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/random_esd_church_scissorhands"}
    )

random_esd_church_diffuser_config = RandomESDChurchConfigDiffuser()
