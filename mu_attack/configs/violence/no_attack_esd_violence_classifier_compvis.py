# mu_attack/configs/violence/no_attack_esd_violence_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class NoAttackESDViolenceClassifierConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="no_attack",
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
        no_attack = {
            "dataset_path": "Output/data/Violence"
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/no_attack_esd_violence"}
    )

hard_prompt_esd_violence_classifier_compvis_config = NoAttackESDViolenceClassifierConfigCompvis()


