# mu_attack/configs/nudity/text_grad_esd_nudity_classifier_diffuser.py

from mu_attack.core.base_config import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig


class TextGradESDNudityClassifierDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="diffusers"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        iteration = 1,
        text_grad = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/text_grad_esd_nudity_classifier_scissorhands", 
              "name": "TextGradNudity"}
    )

text_grad_esd_nudity_classifier_diffuser_config = TextGradESDNudityClassifierDiffuser()
