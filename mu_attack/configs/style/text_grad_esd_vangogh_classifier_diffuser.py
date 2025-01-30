# mu_attack/configs/style/text_grad_esd_vangogh_classifier_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class TextGradESDVangoghClassifierConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "vangogh",
        classifier_dir="results/checkpoint-2800",
        backend="diffusers",
        target_ckpt = "files/pretrained/SD-1-4/ESD_ckpt/VanGogh-ESDx1-UNET-SD.pt"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        k = 3,
        text_grad = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/text_grad_esd_vangogh_classifier"}
    )

text_grad_esd_vangogh_classifier_diffuser_config = TextGradESDVangoghClassifierConfigDiffusers()
