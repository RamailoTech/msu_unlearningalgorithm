# mu_attack/configs/violence/hard_prompt_esd_violence_P4D_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDViolenceP4DConfigDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        concept = "harm",
        classifier_dir=None,
        backend="diffusers",
        target_ckpt = "files/pretrained/SD-1-4/ESD_ckpt/Violence-ESDu1-UNET-SD.pt"
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

hard_prompt_esd_violence_P4D_diffuser_config = HardPromptESDViolenceP4DConfigDiffuser()
