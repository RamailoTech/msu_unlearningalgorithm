# mu_attack/configs/object/hard_prompt_esd_church_P4D_diffuser.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDChurchP4DConfigDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        concept = "church",
        target_ckpt = "files/pretrained/SD-1-4/ESD_ckpt/Church-ESDu1-UNET-SD.pt",
        backend="diffusers"
    )

    attacker: AttackerConfig = AttackerConfig(
        k = 3,
        sequential = True,
        hard_prompt = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_church_P4D_semipermeable_membrane"}
    )

hard_prompt_esd_church_P4D_diffuser_config = HardPromptESDChurchP4DConfigDiffuser()
