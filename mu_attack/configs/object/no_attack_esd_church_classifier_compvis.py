# mu_attack/configs/object/no_attack_esd_church_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class NoAttackESDChurchClassifierConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="no_attack",
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
        attack_idx = 1,
        no_attack = {
            "dataset_path": "files/dataset/church"
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/no_attack_esd_church_scissorhands"}
    )

no_attack_esd_church_classifier_compvis_config = NoAttackESDChurchClassifierConfigCompvis()
