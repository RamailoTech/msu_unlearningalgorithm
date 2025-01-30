# mu_attack/configs/violence/no_attack_esd_violence_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class NoAttackESDViolenceClassifierConfigDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="no_attack",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "harm",
        classifier_dir=None,
        backend="diffusers",
        target_ckpt= "files/pretrained/SD-1-4/ESD_ckpt/Violence-ESDu1-UNET-SD.pt",
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        no_attack = {
            "dataset_path": "Output/data/Violence"
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/no_attack_esd_violence"}
    )

no_attack_esd_violence_classifier_diffuser_config = NoAttackESDViolenceClassifierConfigDiffuser()


