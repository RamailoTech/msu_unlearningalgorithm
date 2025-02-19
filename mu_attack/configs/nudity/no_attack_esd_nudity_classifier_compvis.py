# mu_attack/configs/nudity/no_attack_esd_nudity_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig


class NoAttackESDNudityClassifierConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="no_attack",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        sampling_step_num=1,
        criterion="l1",
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        iteration=1,
        attack_idx=1,
        no_attack = {
            "dataset_path": "outputs/dataset/i2p_nude"
        }
    )
    
    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/no_attack_esd_nudity_esd", "name": "NoAttackEsdNudity"}
    )


no_attack_esd_nudity_classifier_compvis_config = NoAttackESDNudityClassifierConfigCompvis()
