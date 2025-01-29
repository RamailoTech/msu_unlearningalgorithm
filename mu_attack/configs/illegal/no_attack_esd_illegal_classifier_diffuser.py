# mu_attack/configs/illegal/no_attack_esd_illegal_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class NoAttackESDIllegalClassifierConfigDiffusers(BaseConfig):
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
        backend="diffusers",
        target_ckpt= "files/pretrained/SD-1-4/ESD_ckpt/Illegal_activity-ESDu1-UNET-SD.pt"
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

no_attack_esd_illegal_classifier_diffusers_config = NoAttackESDIllegalClassifierConfigDiffusers()
