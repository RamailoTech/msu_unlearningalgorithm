# mu_attack/configs/violence/hard_prompt_esd_violence_P4D_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDViolenceP4DConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "harm",
        classifier_dir=None,
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        hard_prompt = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_violence_P4D"}
    )

hard_prompt_esd_violence_P4D_compvis_config = HardPromptESDViolenceP4DConfigCompvis()
