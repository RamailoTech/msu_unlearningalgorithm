# mu_attack/configs/object/text_grad_esd_church_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class TextGradESDChurchClassifierConfigDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
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
        sequential = True,
        text_grad = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/text_grad_esd_church_scissorhands"}
    )

text_grad_esd_church_classifier_diffuser_config = TextGradESDChurchClassifierConfigDiffuser ()
