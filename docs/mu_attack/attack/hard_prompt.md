
## UnlearnDiffAttak

This repository contains the implementation of UnlearnDiffAttack for hard prompt attack, a framework for evaluating the robustness of safety-driven unlearned Models using adversarial prompts.


### Create Environment 

```
create_env <algorithm_name>
```
eg: ```create_env mu_attack```

```
conda activate <environment_name>
```
eg: ```conda activate mu_attack```



### Generate Dataset
```
python -m scripts.generate_dataset --prompts_path data/prompts/prompts.csv --concept i2p_nude --save_path outputs/dataset
```



### Run Attack 
1. **Hard Prompt Attack - compvis**

```python
from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
from mu_attack.execs.attack import MUAttack
from mu.algorithms.scissorhands.configs import scissorhands_train_mu

def run_attack_for_nudity():

    overridable_params = {
        "task.compvis_ckpt_path":"outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "task.compvis_config_path": scissorhands_train_mu.model_config_path,
        "task.dataset_path":"outputs/dataset/i2p_nude",
        "logger.json.root":"results/hard_prompt_esd_nudity_P4D_scissorhands",
    }

    MUAttack(
        config=hard_prompt_esd_nudity_P4D_compvis_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_attack_for_nudity()
```


**Code Explanation & Important Notes**

1. from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_compvis_config
→ This imports the predefined Hard Prompt Attack configuration for nudity unlearning in the CompVis model. It sets up the attack parameters and methodologies.

2. from mu.algorithms.scissorhands.configs import scissorhands_train_mu
→ Imports the Scissorhands model configuration, required to set the task.compvis_config_path parameter correctly.

**Overriding Parameters in JSON Configuration**

* The overridable_params dictionary allows dynamic modification of parameters defined in the JSON configuration.

* This enables users to override default values by passing them as arguments.

**Example usage**

```python
overridable_params = {
    "task.compvis_ckpt_path": "outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
    "task.compvis_config_path": scissorhands_train_mu.model_config_path,  # Overrides model config
    "task.dataset_path": "outputs/dataset/i2p_nude",  # Overrides dataset path
    "logger.json.root": "results/hard_prompt_esd_nudity_P4D_scissorhands",  # Overrides logging path
    "attacker.k" = 3,
}

```


2. **Hard Prompt Attack - diffuser**

```python
from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_diffusers_config
from mu_attack.execs.attack import MUAttack

def run_attack_for_nudity():

    overridable_params = {
       "task.diffusers_model_name_or_path" : "outputs/forget_me_not/finetuned_models/Abstractionism",
        "task.dataset_path" : "outputs/dataset/i2p_nude",
        "logger.json.root" :"results/hard_prompt_esd_nudity_P4D_abstractionism"
    }

    MUAttack(
        config=hard_prompt_esd_nudity_P4D_diffusers_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_attack_for_nudity()
```


**Code Explanation & Important Notes**

1. from mu_attack.configs.nudity import hard_prompt_esd_nudity_P4D_diffusers_config
→ This imports the predefined Hard Prompt Attack configuration for nudity unlearning in the diffusers model. It sets up the attack parameters and methodologies.


### Description of fields in config json file

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

    ```json
    "json": {
            "root": "results/no_attack_esd_nudity_esd",
            "name": "P4D"
        }
    ```


