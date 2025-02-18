# mu_attack/configs/illegal/no_attack_esd_illegal_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class NoAttackESDIllegalClassifierConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="no_attack",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "harm",
        criterion = "l2",
        classifier_dir=None,
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        no_attack = {
            "dataset_path": "files/dataset/illegal"
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "files/results/no_attack_esd_illegal"}
    )

no_attack_esd_illegal_classifier_compvis_config = NoAttackESDIllegalClassifierConfigCompvis()
