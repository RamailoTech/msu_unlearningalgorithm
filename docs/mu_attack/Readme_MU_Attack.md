
## UnlearnDiffAttak

This repository contains the implementation of UnlearnDiffAtk, a framework for evaluating the robustness of safety-driven unlearned Diffusion Models using adversarial prompts.


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


#### Optional(Create environment for specific algorithm):

If you want to create algorithm specific environment then use command given below:

```bash
create_env erase_diff
```

The <algorithm_name> has to be one of the folders in the `mu/algorithms` folder.

### Generate Dataset
```
python -m scripts.generate_dataset --prompts_path data/prompts/prompts.csv --concept i2p_nude --save_path outputs/dataset
```


### Description of fields in config json file for diffuser

1. overall

This section defines the high-level configuration for the attack.

* task : The name of the task being performed.

    Type: str
    Example: P4D, classifer

* attacker: Specifies the attack type.

    Type: str
    Example: hard_prompt, no_attack

* logger: Defines the logging mechanism.

    Type: str
    Example: JSON

* resume: Option to resume from previous checkpoint.


2. task


* concept: The concept targeted by the attack.

    Type: str
    Example: nudity, harm

* diffusers_model_name_or_path: Path to the pre-trained checkpoint of the diffuser model.

    Type: str
    Example: "outputs/semipermeable_membrane/finetuned_models/"


* target_ckpt: Path to the target model checkpoint used in the attack.

    Type: str
    Example: "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt"

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

* backend: Specifies the backend model i.e "diffuser".


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


### Description of fields in config json file for compvis

1. overall

This section defines the high-level configuration for the attack.

* task : The name of the task being performed.

    Type: str
    Example: P4D, classifer

* attacker: Specifies the attack type.

    Type: str
    Example: hard_prompt, no_attack

* logger: Defines the logging mechanism.

    Type: str
    Example: JSON

* resume: Option to resume from previous checkpoint.


2. task


* concept: The concept targeted by the attack.

    Type: str
    Example: nudity, harm

* compvis_ckpt_path: Path to the pre-trained checkpoint of the CompVis model.

    Type: str
    Example: "outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth"


* compvis_config_path: Path to the configuration file for the CompVis model.

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

* backend: Specifies the backend model i.e "compvis".


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

### Run Attack 
1. Hard Prompt Attack

* compvis

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/hard_prompt_esd_nudity_P4D_compvis.json
```

* diffuser

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/hard_prompt_esd_nudity_P4D_diffuser.json
```


2. Random Attack

* compvis

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/random_esd_nudity_compvis.json
```

* diffuser

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/random_esd_nudity_diffuser.json
```



3. Seed Search

* compvis

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/seed_search_esd_nudity_classifier_compvis.json
```

* diffusers

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/seed_search_esd_nudity_classifier_diffuser.json
```

4. Text Grad

* compvis

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/text_grad_esd_nudity_classifier_compvis.json
```

* diffusers

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/text_grad_esd_nudity_classifier_diffuser.json
```


### Run No Attack

* compvis

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/no_attack_esd_nudity_classifier_compvis.json
```

* diffusers

```bash
python -m mu_attack.execs.attack --config_path mu_attack/configs/nudity/no_attack_esd_nudity_classifier_diffuser.json
```


### Mass Attack
Attack using all attack ids, for a specific attack method.

* Run the following command to make the script executable:

```bash
chmod +x scripts/mass_attack.sh
```

* Run the Script Execute the mass attack script using:

```bash
scripts/mass_attack.sh mu_attack/configs/nudity/hard_prompt_esd_nudity_P4D.json
```



### Evaluation:

In this section, we assess the performance and robustness of the results generated by the attack algorithms

#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file.  

**Example Code**

```python
from mu_attack.evaluators.asr import ASREvaluator
from mu_attack.evaluators.clip_score import ClipScoreEvaluator
from mu_attack.evaluators.fid import FIDEvaluator
from mu_attack.configs.evaluation import attack_evaluation_config


def main():
    # Initialize the configuration
    config = attack_evaluation_config
    config.asr.root = "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/results/hard_prompt_esd_nudity_P4D_concept_ablation/P4d"
    config.asr.root_no_attack = "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/results/no_attack_esd_nudity/NoAttackEsdNudity"
    config.clip.devices = "0"
    config.clip.image_path = "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/results/hard_prompt_esd_nudity_P4D_concept_ablation/P4d/images"
    config.clip.log_path = "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/results/hard_prompt_esd_nudity_P4D_concept_ablation/P4d/log.json"
    config.fid.ref_batch_path = "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/results/hard_prompt_esd_nudity_P4D_concept_ablation/P4d/images"
    config.fid.sample_batch_path = "/home/ubuntu/Projects/balaram/unlearn_diff_attack/outputs/dataset/i2p_nude/imgs"

    # Common output path
    config.output_path = "/home/ubuntu/Projects/Palistha/unlearn_diff_attack/results/evaluation/results.json"

    # Initialize and run the ASR evaluator
    asr_evaluator = ASREvaluator(
        config = attack_evaluation_config,
        root=config.asr.root,
        root_no_attack=config.asr.root_no_attack,
        output_path=config.output_path
    )
    print("Running ASR Evaluator...")
    asr_evaluator.run()

    # Initialize and run the CLIP Score evaluator
    clip_evaluator = ClipScoreEvaluator(
        config = attack_evaluation_config,
        image_path=config.clip.image_path,
        log_path=config.clip.log_path,
        output_path=config.output_path,
        devices = config.clip.devices
    )
    print("Running CLIP Score Evaluator...")
    clip_evaluator.run()

    # Initialize and run the FID evaluator
    fid_evaluator = FIDEvaluator(
        config = attack_evaluation_config,
        ref_batch_path=config.fid.ref_batch_path,
        sample_batch_path=config.fid.sample_batch_path,
        output_path=config.output_path
    )
    print("Running FID Evaluator...")
    fid_evaluator.run()


if __name__ == "__main__":
    main()
]

```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```

**How It Works** 
* Default Values: The script first loads default values from the evluation config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the evaluation. 


**Evaluation Metrics:**

* Attack Succes Rate (ASR)

* Fréchet inception distance(FID): evaluate distributional quality of image generations, lower is better.

* CLIP score : measure contextual alignment with prompt descriptions, higher is better.


**Configuration File Structure for Evaluator**

* ASR Evaluator Configuration

    - root: Directory containing results with attack.
    - root-no-attack: Directory containing results without attack.

* Clip Evaluator Configuration

    - image_path: Path to the directory containing generated images to evaluate.
    - devices: Device ID(s) to use for evaluation. Example: "0" for the first GPU or "0,1" for multiple GPUs.
    - log_path: Path to the log file containing prompt for the generated images.
    - model_name_or_path: Path or model name for the pre-trained CLIP model. Default is "openai/clip-vit-base-patch32".

* FID Evaluator Configuration

    - ref_batch_path: Path to the directory containing reference images.
    - sample_batch_path: Path to the directory containing generated/sample images.

* Global Configuration

    - output_path: Path to save the evaluation results as a JSON file.


