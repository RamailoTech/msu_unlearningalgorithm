
## UnlearnDiffAttak

This repository contains the implementation of UnlearnDiffAttack for text grad, a framework for evaluating the robustness of safety-driven unlearned Models using adversarial prompts.


## Usage

This section contains the usage guide for the package.

### Installation

#### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

Before installing the unlearn_diff package, follow these steps to set up your environment correctly. These instructions ensure compatibility with the required dependencies, including Python, PyTorch, and ONNX Runtime.


**Step-by-Step Setup:**

Step 1. Create a Conda Environment Create a new Conda environment named myenv with Python 3.8.5:

```bash
conda create -n myenv python=3.8.5
```

Step 2. Activate the Environment Activate the environment to work within it:

```bash
conda activate myenv
```

Step 3. Install Core Dependencies Install PyTorch, torchvision, CUDA Toolkit, and ONNX Runtime with specific versions:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 onnxruntime==1.16.3 -c pytorch -c conda-forge
```

Step 4. Install our unlearn_diff Package using pip:

```bash
pip install unlearn_diff
```

Step 5. Install Additional Git Dependencies:

 After installing unlearn_diff, install the following Git-based dependencies in the same Conda environment to ensure full functionality:

```bash
pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```

```bash
pip install git+https://github.com/openai/CLIP.git@main#egg=clip
```

```bash
pip install git+https://github.com/crowsonkb/k-diffusion.git
```

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

```bash
pip install git+https://github.com/Phoveran/fastargs.git@main#egg=fastargs
```

```bash
pip install git+https://github.com/boomb0om/text2image-benchmark
```


### Generate Dataset

Before running attacks you need to generate dataset. Run the following command into the terminal.

```bash
generate_attack_dataset --prompts_path data/prompts/nudity_sample.csv --concept i2p_nude --save_path outputs/dataset --num_samples 1
```

Note: If you want to generate image using full prompt then use `data/prompts/nudity.csv` as prompts_path.



### Run Attack 

1. **Text Grad Attack - compvis**

Use the following code if you wish to run the seed search attack using the CompVis model directly (without converting it into Diffusers format):

```python
from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
from mu_attack.execs.attack import MUAttack
from mu.algorithms.scissorhands.configs import scissorhands_train_mu

def run_attack_for_nudity():

    overridable_params = {
    "task.compvis_ckpt_path" :"/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
    "task.compvis_config_path" : scissorhands_train_mu.model_config_path,
    "task.dataset_path" : "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/outputs/dataset/i2p_nude",
    "attacker.text_grad.lr": 0.02,
    "logger.json.root" : "results/seed_search_esd_nudity_P4D_scissorhands"

    }

    MUAttack(
        config=text_grad_esd_nudity_classifier_compvis_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_attack_for_nudity()
```

2.  **Text Grad Attack – CompVis to Diffusers Conversion**

If you want to convert the CompVis model into the Diffusers format before running the attack, use the following code. Note: For the conversion to take place, set task.save_diffuser to True and to use the converted model task.sld should be set to None.

```python
from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
from mu_attack.execs.attack import MUAttack
from mu.algorithms.scissorhands.configs import scissorhands_train_mu

def run_attack_for_nudity():

    overridable_params = {
        "task.compvis_ckpt_path" :"/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "task.compvis_config_path" : scissorhands_train_mu.model_config_path,
        "task.dataset_path" : "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/outputs/dataset/i2p_nude",
        "attacker.text_grad.lr": 0.02,
        "logger.json.root" : "results/seed_search_esd_nudity_P4D_scissorhands",
        "task.save_diffuser": True, # This flag triggers conversion
        "task.sld": None, # Set sld to None for conversion
        "task.model_name": "SD-v1-4"
    }

    MUAttack(
        config=text_grad_esd_nudity_classifier_compvis_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_attack_for_nudity()
```

**For Conversion:**

When converting a CompVis model to the Diffusers format, ensure that task.save_diffuser is set to True and task.sld is set to None. This instructs the pipeline to perform the conversion during initialization and then load the converted checkpoint.

**Code Explanation & Important Notes**

1. from mu_attack.configs.nudity import text_grad_esd_nudity_P4D_compvis_config
→ This imports the predefined text grad Attack configuration for nudity unlearning in the CompVis model. It sets up the attack parameters and methodologies.

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
    "logger.json.root": "results/seed_search_esd_nudity_P4D_scissorhands",  # Overrides logging path
    "attacker.k" = 3,
    "attacker.no_attack.dataset_path" = "path/to/dataset" #overrides the datset path for no attack
}

```

2. **Text Grad Attack - diffuser**

```python
from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_diffuser_config
from mu_attack.execs.attack import MUAttack

def run_attack_for_nudity():

    overridable_params = {
    "task.diffusers_model_name_or_path" : "outputs/forget_me_not/finetuned_models/Abstractionism",
    "task.dataset_path" : "outputs/dataset/i2p_nude",
    "logger.json.root" : "results/random_esd_nudity_diffuser_uce"

    }

    MUAttack(
        config=text_grad_esd_nudity_classifier_diffuser_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_attack_for_nudity()
```


**Code Explanation & Important Notes**

1. from mu_attack.configs.nudity import text_grad_esd_nudity_P4D_diffusers_config
→ This imports the predefined Text Grad Attack configuration for nudity unlearning in the diffusers model. It sets up the attack parameters and methodologies.


### Description of fields in config json file

1. overall

This section defines the high-level configuration for the attack.

* task : The name of the task being performed.

    Type: str
    Example: classifer

* attacker: Specifies the attack type.

    Type: str
    Example: text_grad

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

* text_grad: Json that contains lr and weight_decay.

    Type: Json
    Example: 
    ```json
            "text_grad": {
            "lr": 0.01,
            "weight_decay": 0.1
        }
    ```


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
        Example: "results/seed_search_esd_nudity_P4D"


    - name: Name for the log file or experiment.

        - Type: str
        - Example: "Seed Search Nudity"

    Example usage:

        "json": {
                "root": "results/text_grad_esd_nudity_esd",
                "name": "TextGradNudity"
            }



