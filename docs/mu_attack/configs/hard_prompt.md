
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

### Description of fields in config file

1. overall

This section defines the high-level configuration for the attack.

* task : The name of the task being performed.

    Type: str
    Example:  classifer

* attacker: Specifies the attack type.

    Type: str
    Example: no_attack

* logger: Defines the logging mechanism.

    Type: str
    Example: JSON

* resume: Option to resume from previous checkpoint.


2. task


* concept: The concept targeted by the attack.

    Type: str
    Example: nudity, harm

* diffusers_model_name_or_path: Path to the pre-trained checkpoint of the diffuser model. (For diffuser)

    Type: str
    Example: "outputs/semipermeable_membrane/finetuned_models/"


* target_ckpt: Path to the target model checkpoint used in the attack.  (For diffuser)

    Type: str
    Example: "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt"


* compvis_ckpt_path: Path to the pre-trained checkpoint of the CompVis model. (For compvis)

    Type: str
    Example: "outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth"


* compvis_config_path: Path to the configuration file for the CompVis model. (For compvis)

    Type: str
    Example: "configs/scissorhands/model_config.yaml"

* cache_path: Directory to cache intermediate results.

    Type: str
    Example: ".cache"

* dataset_path: Path to the dataset used for the attack.

    Type: str
    Example: "outputs/dataset/i2p_nude"

* criterion: The loss function or criterion used during the attack.

    Type: str
    Example: "l2"

* classifier_dir: Directory for the classifier, if applicable. null if not used.
    Type: str
    Example: "/path/classifier_dir"

* sampling_step_num: Number of sampling steps during the attack.

    Type: int
    Example: 1

* sld: Strength of latent disentanglement.

    Type: str
    Example: "weak" 

* sld_concept: Concept tied to latent disentanglement.

    Type: str
    Example: "nudity"

* negative_prompt: The negative prompt used to steer the generation. 

    Type: str
    Example: ""

* model_name: Name of the model. The model_name parameter determines which base Stable Diffusion model is used by the pipeline.

    Type: str
    Example: "SD-v1-4"
    Choices: "SD-v1-4", "SD-V2", "SD-V2-1"

* save_diffuser: A Boolean flag that determines whether the CompVis model should be converted into the Diffusers format before being used.

    Type: str
    Example: True

    Behavior:
    * If set to True, the pipeline will perform a conversion of the CompVis model into the Diffusers format and then load the converted checkpoint.

    * If set to False, the conversion is skipped and the model remains in its original CompVis format for use and uses compvis based implementation.

* converted_model_folder_path: Folder path to save the converted compvis model to diffuser.

    Type: str
    Example: "outputs"

* backend: Specifies the backend model i.e "diffusers".

    Type: str
    Options: "diffusers" or "compvis"


3. attacker

* insertion_location: The point of insertion for the prompt.

    Type: str
    Example: "prefix_k"

* k: The value of k for the prompt insertion point.

    Type: int
    Example: 5

* iteration: Number of iterations for the attack.

    Type: int
    Example: 1

* seed_iteration: Random seed for the iterative process.

    Type: int
    Example: 1

* attack_idx: Index of the attack for evaluation purposes.

    Type: int
    Example: 0

* eval_seed: Seed value used for evaluation.

    Type: int
    Example: 0

* universal: Whether the attack is universal (true or false).

    Type: bool
    Example: false

* sequential: Whether the attack is applied sequentially.

    Type: bool
    Example: true

* lr: Learning rate for the attack optimization process.

    Type: float
    Example: 0.01

* weight_decay: Weight decay applied during optimization.

    Type: float
    Example: 0.1

4. logger

* json: Logging configuration.

    - root: Path to the directory where logs will be saved.

        Type: str
        Example: "results/hard_prompt_esd_nudity_P4D"


* name: Name for the log file or experiment.

    - Type: str
    - Example: "P4D"

    Example usage:

        {
        "json": {
            "root": "results/no_attack_esd_nudity_esd",
            "name": "P4D"
            }
        }
