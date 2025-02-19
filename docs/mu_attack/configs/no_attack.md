**Sample no attack config for compvis**


```python
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

```


**Sample compvis config json for no attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "no_attack",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "compvis_ckpt_path": "outputs/esd/esd_Abstractionism_model.pth",
        "compvis_config_path":"mu/algorithms/esd/configs/model_config.yaml",
        "cache_path": ".cache",
        "dataset_path": "outputs/dataset/i2p_nude",
        "criterion": "l1",
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "sth",
        "backend":"compvis"
    },
    "attacker": {
        "insertion_location": "prefix_k",
        "k": 5,
        "seed_iteration": 1,
        "sequential": true,
        "iteration": 40,
        "attack_idx": 1,
        "eval_seed": 0,
        "universal": false,
        "no_attack": {
            "dataset_path": "outputs/dataset/i2p_nude"
        }
    },
    "logger": {
        "json": {
            "root": "results/no_attack_esd_nudity_esd",
            "name": "NoAttackEsdNudity"
        }
    }
}
```


**Sample config for no attack for diffuser**


```python
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
```


**Sample diffusers config json for no attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "no_attack",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "diffusers_model_name_or_path": "outputs/uce/finetuned_models/uce_Abstractionism_model",
        "target_ckpt": "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
        "cache_path": ".cache",
        "dataset_path": "outputs/dataset/i2p_nude",
        "criterion": "l1",
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "sth",
        "backend":"diffusers"

    },
    "attacker": {
        "insertion_location": "prefix_k",
        "k": 5,
        "seed_iteration": 1,
        "sequential": true,
        "iteration": 40,
        "attack_idx": 1,
        "eval_seed": 0,
        "universal": false,
        "no_attack": {
            "dataset_path": "outputs/dataset/i2p_nude"
        }
    },
    "logger": {
        "json": {
            "root": "results/no_attack_esd_nudity_uce",
            "name": "NoAttackEsdNudity"
        }
    }
}
```

