
## UnlearnDiffAttak

This project implements a novel adversarial unlearning framework designed to perform soft prompt attacks on diffusion models. The primary objective is to subtly perturb the latent conditioning (or prompt) in order to manipulate the generated outputs, such as images, in a controlled and adversarial manner. 


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

```bash
python -m scripts.generate_dataset --prompts_path data/prompts/prompts.csv --concept i2p_nude --save_path outputs/dataset
```


### Run Soft Prompt Attack 
1. **Soft Prompt Attack - compvis**

```python

from mu_attack.execs.adv_attack import AdvAttack
from mu_attack.configs.adv_unlearn import adv_attack_config
from mu.algorithms.esd.configs import esd_train_mu


def mu_defense():
    adv_unlearn = AdvAttack(
        config=adv_attack_config,
        compvis_ckpt_path = "/home/ubuntu/Projects/dipesh/unlearn_diff/models/sd-v1-4-full-ema.ckpt",
        attack_step = 2,
        backend = "compvis",
        config_path = esd_train_mu.model_config_path

    )
    adv_unlearn.attack()

if __name__ == "__main__":
    mu_defense()

```


2. **Soft Prompt Attack - diffuser**

```python
from mu_attack.execs.adv_attack import AdvAttack
from mu_attack.configs.adv_unlearn import adv_attack_config


def mu_defense():

    adv_unlearn = AdvAttack(
        config=adv_attack_config,
        diffusers_model_name_or_path = "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50",
        attack_step = 2,
        backend = "diffusers"

    )
    adv_unlearn.attack()

if __name__ == "__main__":
    mu_defense()

```

**Run the python file in offline mode**

```bash
WANDB_MODE=offline python_file.py
```


**Code Explanation & Important Notes**

* from mu_attack.configs.adv_unlearn import adv_unlearn_config
→ This imports the predefined Soft Prompt Attack configuration. It sets up the attack parameters and methodologies.


**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 


### Description of fields in soft prompt attack config

1. Model setup

* config_path : Path to the inference configuration file for Stable Diffusion v1.4.

    * Type: str
    * Default: "model_config.yaml"

* compvis_ckpt_path : Path to the Stable Diffusion v1.4 checkpoint file.

    * Type: str
    * Default: "models/sd-v1-4-full-ema.ckpt"

* encoder_model_name_or_path : Path to the pre-trained encoder model used for text-to-image training.

    * Type: str
    * Default: "CompVis/stable-diffusion-v1-4"

* diffusers_model_name_or_path : Path to the Diffusers-based implementation of the model.

    * Type: str
    * Default: "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50"

* target_ckpt : Checkpoint path for sampling. If None, it uses the default model.

    * Type: str
    * Default: None

2. Devices & I/O

* devices : Specifies the CUDA devices used for training.

    * Type: str
    * Default: "0,0"

* seperator : Defines the separator used when processing multiple words for unlearning.

    * Type: str
    * Default: None

* cache_path : Path where intermediate results and cache files are stored.

    * Type: str
    * Default: ".cache"


3. Image & Diffusion Sampling

* start_guidance : Guidance scale used for generating the initial image.

    * Type: float
    * Default: 3.0

* ddim_steps : Number of DDIM sampling steps used for inference.

    * Type: int
    * Default: 50

* image_size : The resolution of images generated during training.

    * Type: int
    * Default: 512

* ddim_eta : Noise scaling factor for DDIM inference.

    * Type: float
    * Default: 0


* prompt: The text prompt associated with the concept to erase.

    * Type: str
    * Default: "nudity"

* attack_method: The adversarial attack method used during training.

    * Type: str
    * Choices: ["pgd", "multi_pgd", "fast_at", "free_at"]
    * Default: "pgd"

* ddim_eta: The DDIM sampling noise parameter.

    * Type: float
    * Default: 0

5. Adversarial Attack Hyperparameters

* adv_prompt_num: Number of prompt tokens used for adversarial learning.

    * Type: int
    * Default: 1

* attack_embd_type: Type of embedding targeted for attack.

    * Type: str
    * Choices: ["word_embd", "condition_embd"]
    * Default: "word_embd"

* attack_type: The type of attack applied.

    * Type: str
    * Choices: ["replace_k", "add", "prefix_k", "suffix_k", "mid_k", "insert_k", "per_k_words"]
    * Default: "prefix_k"

* attack_init: Method for initializing adversarial attacks.

    * Type: str
    * Choices: ["random", "latest"]
    * Default: "latest"

* attack_step: Number of attack optimization steps.

    * Type: int
    * Default: 30

* attack_lr: Learning rate for adversarial attack updates.

    * Type: float
    * Default: 1e-3


6. Backend & Logging

* backend: Specifies the backend for diffusion-based training.

    * Type: str
    * Default: "diffusers"

* project_name: Name of the WandB project for logging.

    * Type: str
    * Default: "quick-canvas-machine-unlearning"
