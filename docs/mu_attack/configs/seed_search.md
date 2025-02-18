**Sample config for seed search attack for compvis**

```python
# mu_attack/configs/nudity/seed_search_esd_nudity_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class SeedSearchESDNudityClassifierCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="seed_search",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="compvis",
        diffusers_config_file = None,
        save_diffuser = False
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        attack_idx=1,
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/seed_search_esd_nudity_scissorhands", 
              "name": "Seed Search Nudity"}
    )

seed_search_esd_nudity_classifier_compvis_config = SeedSearchESDNudityClassifierCompvis()

```



**Sample compvis config for seed search attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "seed_search",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "compvis_ckpt_path": "outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "compvis_config_path":"mu/algorithms/scissorhands/configs/model_config.yaml",
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
        "iteration": 40,
        "attack_idx": 1,
        "eval_seed": 0,
        "universal": false,
        "sequential": true,
        "seed_iteration": 1
    },
    "logger": {
        "json": {
            "root": "results/seed_search_esd_nudity_scissorhands",
            "name": "Seed Search Nudity"
        }
    }
}
```


**Sample config for seed search diffuser**

```python
class SeedSearchESDNudityClassifierDiffusers(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="seed_search",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="diffusers",
        target_ckpt= "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        attack_idx=1,
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/seed_search_esd_nudity_scissorhands", 
              "name": "Seed Search Nudity"}
    )
```

**Sample diffusers config for seed search attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "seed_search",
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
        "iteration": 40,
        "attack_idx": 1,
        "eval_seed": 0,
        "universal": false,
        "sequential": true,
        "seed_iteration": 1
    },
    "logger": {
        "json": {
            "root": "results/seed_search_esd_nudity_uce",
            "name": "Seed Search Nudity"
        }
    }
}
```