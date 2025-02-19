**Sample config for text grad for compvis**

```python
# mu_attack/configs/nudity/text_grad_esd_nudity_classifier_compvis.py

from mu_attack.core import BaseConfig, OverallConfig, TaskConfig, AttackerConfig, LoggerConfig

class TextGradESDNudityClassifierCompvis(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
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
        iteration = 1,
        text_grad = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/text_grad_esd_nudity_classifier_scissorhands", 
              "name": "TextGradNudity"}
    )

text_grad_esd_nudity_classifier_compvis_config = TextGradESDNudityClassifierCompvis()

```


**Sample compvis config for text grad attack**

```json
{
    "overall": {
        "task": "classifier",
        "attacker": "text_grad",
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
        "negative_prompt": "sth",
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
        "text_grad": {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    },
    "logger": {
        "json": {
            "root": "results/text_grad_esd_nudity_classifier_scissorhands",
            "name": "TextGradNudity"
        }
    }
}
```


**Sample config for text grad for diffusers**

```python
class TextGradESDNudityClassifierDiffuser(BaseConfig):
    overall: OverallConfig = OverallConfig(
        task="classifier",
        attacker="text_grad",
        logger="json",
        resume=None
    )

    task: TaskConfig = TaskConfig(
        sampling_step_num=1,
        sld="weak",
        sld_concept="nudity",
        negative_prompt="sth",
        backend="diffusers"
    )

    attacker: AttackerConfig = AttackerConfig(
        sequential=True,
        iteration = 1,
        text_grad = {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    )

    logger: LoggerConfig = LoggerConfig(
        json={"root": "results/text_grad_esd_nudity_classifier_scissorhands", 
              "name": "TextGradNudity"}
    )
```


**Sample diffusers config for text grad attack**

```json

{
    "overall": {
        "task": "classifier",
        "attacker": "text_grad",
        "logger": "json",
        "resume": null
    },
    "task": {
        "concept": "nudity",
        "diffusers_model_name_or_path": "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50",
        "target_ckpt": "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
        "cache_path": ".cache",
        "dataset_path": "outputs/dataset/i2p_nude",
        "criterion": "l2",
        "classifier_dir": null,
        "sampling_step_num": 1,
        "sld": "weak",
        "sld_concept": "nudity",
        "negative_prompt": "sth",
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
        "text_grad": {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    },
    "logger": {
        "json": {
            "root": "results/text_grad_esd_nudity_classifier_uce",
            "name": "TextGradNudity"
        }
    }
}
```