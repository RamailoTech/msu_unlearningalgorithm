# mu_attack/configs/nudity/no_attack_esd_nudity_classifier_diffuser.py

from mu_attack.core.base_config import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig


class NoAttackESDNudityClassifierConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="no_attack",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        sampling_step_num=1,
        target_ckpt= "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
        criterion="l1",
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="diffusers"
    )

    attacker: AttackerConfig = AttackerConfig(
        iteration=1,
        no_attack = {
            "dataset_path": "outputs/dataset/i2p_nude"
        }
    )
    
    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/no_attack_esd_nudity_esd", "name": "NoAttackEsdNudity"}
    )

no_attack_esd_nudity_classifier_diffusers_config = NoAttackESDNudityClassifierConfigDiffusers()
