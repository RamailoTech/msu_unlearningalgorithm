
**Sample hard_prompt_config for compvis**

```python
# mu_attack/configs/nudity/hard_prompt_esd_nudity_P4D_compvis.py

import os
from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class HardPromptESDNudityP4DConfigCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False,
        converted_model_folder_path = "outputs"

    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        lr=0.01,
        weight_decay=0.1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/hard_prompt_esd_nudity_P4D_scissorhands", "name": "P4d"}
    )

hard_prompt_esd_nudity_P4D_compvis_config = HardPromptESDNudityP4DConfigCompvis()

```


**Sample hard_prompt config json for compvis**

```json
{
    "overall": {
        "task": "P4D",
        "attacker": "hard_prompt",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "compvis_ckpt_path": "outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "compvis_config_path":"mu/algorithms/scissorhands/configs/model_config.yaml",
        "cache_path": ".cache",
        "dataset_path": "outputs/dataset/i2p_nude",
        "criterion": "l2",
        "classifier_dir": null,
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "",
        "backend":"compvis"
    },
    "attacker": {
        "insertion_location": "prefix_k",
        "k": 5,
        "iteration": 1,
        "seed_iteration": 1,
        "attack_idx": 0,
        "eval_seed": 0,
        "universal": false,
        "sequential": true,
        "lr": 0.01,
        "weight_decay": 0.1
    },
    "logger": {
        "json": {
            "root": "results/hard_prompt_esd_nudity_P4D_scissorhands",
            "name": "P4d"
        }
    }
}
```


**Sample hard_prompt_config for diffusers**

```python
class HardPromptESDNudityP4DConfigDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="P4D",
        attacker="hard_prompt",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        classifier_dir=None,
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="",
        backend="diffusers"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential = True,
        lr=0.01,
        weight_decay=0.1
    )

    logger: LoggerConfig = LoggerConfig(
        json={
            "root": "results/hard_prompt_esd_nudity_P4D_scissorhands",
            "name": "P4d"
            }
    )
```

**Sample hard_prompt config json for diffusers**

```json
{
    "overall": {
        "task": "P4D",
        "attacker": "hard_prompt",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "diffusers_model_name_or_path": "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/forget_me_not/finetuned_models/Abstractionism",
        "target_ckpt": "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
        "cache_path": ".cache",
        "dataset_path": "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/outputs/dataset/i2p_nude",
        "criterion": "l2",
        "classifier_dir": null,
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "",
        "backend":"diffusers"
    },
    "attacker": {
        "insertion_location": "prefix_k",
        "k": 5,
        "iteration": 1,
        "seed_iteration": 1,
        "attack_idx": 0,
        "eval_seed": 0,
        "universal": false,
        "sequential": true,
        "lr": 0.01,
        "weight_decay": 0.1
    },
    "logger": {
        "json": {
            "root": "results/hard_prompt_esd_nudity_P4D_semipermeable_membrane",
            "name": "P4d"
        }
    }
}
```