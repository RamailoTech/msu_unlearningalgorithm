# mu_attack/configs/illegal/text_grad_esd_illegal_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class TextGradESDIllegalClassifierConfigCompvis(BaseConfig):
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
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
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

text_grad_esd_illegal_classifier_compvis_config = TextGradESDIllegalClassifierConfigCompvis()
