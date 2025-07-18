#### No attack
This term serves as a baseline or control condition in adversarial testing. It represents the model's standard behavior when it is not subjected to any specific attack method. This allows researchers to measure the effectiveness and impact of different attack algorithms by comparing the model's performance under attack to its normal, "no attack" operation.

### Generate Dataset

Before running attacks you need to generate dataset. Run the following command in terminal.

```bash
generate_attack_dataset --prompts_path data/prompts/nudity_sample.csv --concept i2p_nude --save_path outputs/dataset --num_samples 1
```

Note: If you want to generate image using full prompt then use `data/prompts/nudity.csv` as prompts_path.

### Run Attack 

**No Attack – CompVis to Diffusers Conversion**

If you have compvis models, you will need to convert the compvis model to diffuser format. Note: For the conversion to take place, set task.`save_diffuser` to True and to use the converted model `task.sld` should be set to None.

```python
from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_compvis_config
from mu_attack.execs.attack import MUAttack
from mu.algorithms.scissorhands.configs import scissorhands_train_mu

def run_attack_for_nudity():

    overridable_params = {
        "task.compvis_ckpt_path" : "outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
        "task.compvis_config_path" : scissorhands_train_mu.model_config_path,
        "task.dataset_path" : "outputs/dataset/i2p_nude",
        "logger.json.root" : "results/no_attack_esd_nudity_P4D_scissorhands",
    "attacker.no_attack.dataset_path" : "outputs/dataset/i2p_nude",
        "task.save_diffuser": True, # This flag triggers conversion
        "task.sld": None, # Set sld to None for conversion
        "task.model_name": "SD-v1-4"
    }

    MUAttack(
        config=no_attack_esd_nudity_classifier_compvis_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_attack_for_nudity()
```


**For Conversion:**

When converting a CompVis model to the Diffusers format, ensure that task.save_diffuser is set to True and task.sld is set to None. This instructs the pipeline to perform the conversion during initialization and then load the converted checkpoint.


**Code Explanation & Important Notes**

1. from mu_attack.configs.nudity import no_attack_esd_nudity_P4D_compvis_config
→ This imports the predefined No Attack configuration for nudity unlearning in the CompVis model. It sets up the attack parameters and methodologies.

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
    "logger.json.root": "results/no_attack_esd_nudity_P4D_scissorhands",  # Overrides logging path
    "attacker.k" = 3,
    "attacker.no_attack.dataset_path" = "path/to/dataset" #overrides the datset path for no attack
}

```

2. **No Attack - diffuser**

```python
from mu_attack.configs.nudity import no_attack_esd_nudity_classifier_diffusers_config
from mu_attack.execs.attack import MUAttack

def run_no_attack_for_nudity():

    overridable_params = {
    "task.diffusers_model_name_or_path" :"outputs/forget_me_not/finetuned_models/Abstractionism",
    "task.dataset_path" : "outputs/dataset/i2p_nude",
    "logger.json.root" : "results/no_attack_esd_nudity_P4D_abstrc",
    "attacker.no_attack.dataset_path" : "outputs/dataset/i2p_nude"
    }

    MUAttack(
        config=no_attack_esd_nudity_classifier_diffusers_config,
        **overridable_params
    )

if __name__ == "__main__":
    run_no_attack_for_nudity()
```


**Code Explanation & Important Notes**

1. from mu_attack.configs.nudity import no_attack_esd_nudity_P4D_diffusers_config
→ This imports the predefined no attack Attack configuration for nudity unlearning in the diffusers model. It sets up the attack parameters and methodologies.


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

fid, _ = fid_score(generated_image_dir=gen_path) #Defaults to the COCO dataset if reference_image_dir is not provided."
print(fid)

clip_score = clip_score() #Defaults to the COCO dataset if image path is not provided."
print(clip_score)

#Optionally provide your own dataset path
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


