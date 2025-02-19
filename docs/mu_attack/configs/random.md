
**Sample config for random attack for compvis**

```python
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


```

**Sample compvis config json for random attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "random",
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
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "sth",
        "backend":"compvis"

    },
    "attacker": {
        "insertion_location": "prefix_k",
        "k": 5,
        "iteration": 40,
        "attack_idx": 1,
        "eval_seed": 0,
        "universal": false,
        "sequential": true,
        "seed_iteration": 1
    },
    "logger": {
        "json": {
            "root": "results/random_esd_nudity_scissorhands",
            "name": "Hard Prompt"
        }
    }
}
```


**Sample config for random attack for diffuser**

```python
class RandomESDNudityDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="random",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="diffusers",
        target_ckpt="files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        attack_idx=1
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/random_esd_nudity_scissorhands", "name": "Hard Prompt"}
    )
```

**Sample diffusers config json for random attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "random",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "diffusers_model_name_or_path": "outputs/uce/finetuned_models/uce_Abstractionism_model",
        "target_ckpt":"files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
        "cache_path": ".cache",
        "dataset_path": "outputs/dataset/i2p_nude",
        "criterion": "l2",
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "sth",
        "backend":"diffusers"
    },
    "attacker": {
        "insertion_location": "prefix_k",
        "k": 5,
        "iteration": 40,
        "attack_idx": 1,
        "eval_seed": 0,
        "universal": false,
        "sequential": true,
        "seed_iteration": 1
    },
    "logger": {
        "json": {
            "root": "results/random_esd_nudity_uce",
            "name": "Hard Prompt"
        }
    }
}
```
