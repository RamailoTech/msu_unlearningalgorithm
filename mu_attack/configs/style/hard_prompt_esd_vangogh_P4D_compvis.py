# mu_attack/configs/style/hard_prompt_esd_vangogh_P4D_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDVangoghP4DConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "vangogh",
        classifier_dir="results/checkpoint-2800",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        k = 3,
        attack_idx = 1,
        hard_prompt = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_vangogh_P4D"}
    )

hard_prompt_esd_vangogh_P4D_compvis_config = HardPromptESDVangoghP4DConfigCompvis()
