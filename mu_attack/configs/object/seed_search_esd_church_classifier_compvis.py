# mu_attack/configs/object/seed_search_esd_church_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class SeedSearchESDChurchClassifierConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="seed_search",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        concept = "church",
        backend="compvis",
        criterion = "l1",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        k = 3,
        attack_idx = 1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/seed_search_esd_church_scissorhands"}
    )

seed_search_esd_church_classifier_compvis_config = SeedSearchESDChurchClassifierConfigCompvis()
