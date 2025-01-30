# mu_attack/configs/object/seed_search_esd_church_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class SeedSearchESDChurchClassifierConfigDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="seed_search",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        concept = "church",
        backend="diffusers",
        criterion = "l1"
    )

    attacker: AttackerConfig = AttackerConfig(
        k = 3,
        attack_idx = 1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/seed_search_esd_church_scissorhands"}
    )

seed_search_esd_church_classifier_diffuser_config = SeedSearchESDChurchClassifierConfigDiffuser()
