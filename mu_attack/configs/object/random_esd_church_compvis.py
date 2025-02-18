# mu_attack/configs/object/random_esd_church_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class RandomESDChurchConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="random",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        concept = "church",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        k = 3,
        attack_idx = 1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/random_esd_church_scissorhands"}
    )

random_esd_church_compvis_config = RandomESDChurchConfigCompvis()
