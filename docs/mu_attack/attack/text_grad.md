### Generate Dataset

Before running attacks you need to generate dataset. Run the following command into the terminal.

```bash
generate_attack_dataset --prompts_path data/prompts/nudity_sample.csv --concept i2p_nude --save_path outputs/dataset --num_samples 1
```

Note: If you want to generate image using full prompt then use `data/prompts/nudity.csv` as prompts_path.



### Run Attack 

**Text Grad Attack – CompVis to Diffusers Conversion**

If you want to convert the CompVis model into the Diffusers format before running the attack, use the following code. Note: For the conversion to take place, set task.save_diffuser to True and to use the converted model task.sld should be set to None.

```python
from mu_attack.configs.nudity import text_grad_esd_nudity_classifier_compvis_config
from mu_attack.execs.attack import MUAttack
from mu.algorithms.scissorhands.configs import scissorhands_train_mu

def run_attack_for_nudity():

    overridable_params = {
        "task.compvis_ckpt_path" :"/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "task.compvis_config_path" : scissorhands_train_mu.model_config_path,
        "task.dataset_path" : "outputs/dataset/i2p_nude",
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


#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the evalautions.  

**Example Code**

```python
from evaluation.metrics.asr import asr_score
from evaluation.metrics.clip import clip_score
from evaluation.metrics.fid import fid_score


root = "results/hard_prompt_esd_nudity_P4D_erase_diff/P4d"
root_no_attack ="results/no_attack_esd_nudity_P4D_abstrctionism/NoAttackEsdNudity"

asr_val = asr_score(root, root_no_attack)
print(asr_val)

images = "results/hard_prompt_esd_nudity_P4D_erase_diff_compvis_to_diffuser/P4d/images"
prompt_path = "results/hard_prompt_esd_nudity_P4D_erase_diff_compvis_to_diffuser/P4d/log.json"
device = "0"
clip_val = clip_score(images, prompt_path, device)

print(clip_val)

gen_path = "results/hard_prompt_esd_nudity_P4D_erase_diff/P4d/images"
ref_path = "data/i2p/nude"
fid_val = fid_score(gen_path,ref_path)
print(fid_val)
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```


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


