# mu_attack/configs/violence/text_grad_esd_violence_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class TextGradESDViolenceClassifierConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
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
        text_grad= {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/text_grad_esd_violence_classifier"}
    )

text_grad_esd_violence_classifier_diffuser_config = TextGradESDViolenceClassifierConfigDiffusers()


