# mu_attack/configs/style/hard_prompt_esd_vangogh_P4D_diffusers.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDVangoghP4DConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
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
        attack_idx = 1,
        hard_prompt = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_vangogh_P4D"}
    )

hard_prompt_esd_vangogh_P4D_diffusers_config = HardPromptESDVangoghP4DConfigDiffusers()
