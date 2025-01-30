#mu_attack/configs/illegal/text_grad_esd_illegal_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class TextGradESDIllegalClassifierConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "harm",
        criterion = "l2",
        classifier_dir=None,
        backend="diffusers",
        target_ckpt = "files/pretrained/SD-1-4/ESD_ckpt/Illegal_activity-ESDu1-UNET-SD.pt",
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        text_grad = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "files/results/text_grad_esd_illegal_classifier_scissorhands"}
    )

text_grad_esd_illegal_classifier_diffusers_config = TextGradESDIllegalClassifierConfigDiffusers()
