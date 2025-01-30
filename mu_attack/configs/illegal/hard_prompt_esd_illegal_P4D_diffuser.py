# mu_attack/configs/illegal/hard_prompt_esd_illegal_P4D_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDIllegalP4DConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "harm",
        criterion = "l2",
        classifier_dir=None,
        backend="diffusers",
        target_ckpt = "files/pretrained/SD-1-4/ESD_ckpt/Illegal_activity-ESDu1-UNET-SD.pt",
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        hard_prompt = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "files/results/hard_prompt_esd_illegal_P4D_scissorhands"}
    )

hard_prompt_esd_illegal_P4D_diffusers_config = HardPromptESDIllegalP4DConfigDiffusers()
