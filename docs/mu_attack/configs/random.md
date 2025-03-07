
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


### Description of fields in config json file

1. overall

This section defines the high-level configuration for the attack.

* task : The name of the task being performed.

    Type: str
    Example: classifer

* attacker: Specifies the attack type.

    Type: str
    Example: random

* logger: Defines the logging mechanism.

    Type: str
    Example: JSON

* resume: Option to resume from previous checkpoint.


2. task


* concept: The concept targeted by the attack.

    Type: str
    Example: nudity

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
    Example: "sth"

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


4. logger

* json: Logging configuration.

    - root: Path to the directory where logs will be saved.

        Type: str
        Example: "results/random_attack_esd_nudity_P4D"


    - name: Name for the log file or experiment.

    - Type: str
    - Example: "random attack"

    Example usage:


        "json": {
                "root": "results/random_attack_esd_nudity_esd",
                "name": "random attack"
            }
