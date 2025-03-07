
## MU_DEFENSE

This repository is for mu_defense that implements adversarial unlearning by integrating a soft prompt attack into the training loop. In this process, a random prompt is selected and its embedding is adversarially perturbed—either at the word or conditional embedding level—to steer the model into unlearning unwanted associations while preserving overall performance.

### Installation
```
pip install unlearn_diff
```

### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

### Create Environment 

```bash
create_env <algorithm_name>

```

**Example**
```bash
create_env mu_defense

```


### Example usage to Run Defense for compvis 

To test the below code snippet, you can create a file, copy the below code in eg, `mu_defense.py` and execute it with `python mu_defense.py` or use `WANDB_MODE=offline python mu_defense.py` for offline mode.

### Run with default config

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config

    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
 
```

### Modify some train parameters in pre defined config class.

View the config descriptions to see a list of available parameters.

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        compvis_ckpt_path = "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/erase_diff/erase_diff_Abstractionism_model.pth",
        attack_step = 2,
        backend = "compvis",
        attack_method = "fast_at",
        warmup_iter = 1,
        iterations = 2,
        model_config_path = erase_diff_train_mu.model_config_path

    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
 
```

**Note: import the model_config_path from the relevant algorithm's configuration module in the mu package**


### Example usage to Run defense for diffusers


### Run with default config

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config
    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
```


### Modify some train parameters in pre defined config class.

View the config descriptions to see a list of available parameters.

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        diffusers_model_name_or_path = "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/forget_me_not/finetuned_models/Abstractionism",
        attack_step = 2,
        backend = "diffusers",
        attack_method = "fast_at",
        warmup_iter = 1,
        iterations = 2
    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
```


### Description of fields in config file

Below is a detailed description of the configuration fields available in the `adv_unlearn_config.py` file. The descriptions match those provided in the help section of the command-line arguments.

1. **Inference & Model Paths**

   * **model_config_path**  
     *Description*: Config path for stable diffusion model. Use for compvis model only. 
     *Type*: `str`  
     *Example*: `configs/stable-diffusion/v1-inference.yaml`

   * **compvis_ckpt_path**  
     *Description*: Checkpoint path for stable diffusion v1-4.  
     *Type*: `str`  
     *Example*: `models/sd-v1-4-full-ema.ckpt`

   * **encoder_model_name_or_path**  
     *Description*: Model name or path for the encoder.
     *Type*: `str`  
     *Example*: `CompVis/stable-diffusion-v1-4`

   * **cache_path**  
     *Description*: Directory used for caching model files.  
     *Type*: `str`  
     *Example*: `.cache`

   * **diffusers_model_name_or_path**  
     *Description*: Model name or path for the diffusers (if used).  
     *Type*: `str`  
     *Example*: `outputs/forget_me_not/finetuned_models/Abstractionism`

   * **target_ckpt**  
     *Description*: Optionally load a target checkpoint into the model for diffuser sampling.  
     *Type*: Typically `str` or `None`  
     *Example*: `path to target checkpoint path`

2. **Devices & IO**

   * **devices**  
     *Description*: CUDA devices to train on.  
     *Type*: `str`  
     *Example*: `0,0`

   * **seperator**  
     *Description*: Separator used if you want to train a bunch of words separately.  
     *Type*: `str` or `None`  
     *Example*: `None`

   * **output_dir**  
     *Description*: Directory where output files (e.g., checkpoints, logs) are saved.  
     *Type*: `str`  
     *Example*: `outputs/adv_unlearn`

3. **Image & Diffusion Sampling**

   * **image_size**  
     *Description*: Image size used to train.  
     *Type*: `int`  
     *Example*: `512`

   * **ddim_steps**  
     *Description*: Number of DDIM steps for inference during training.  
     *Type*: `int`  
     *Example*: `50`

   * **start_guidance**  
     *Description*: Guidance of start image used to train.  
     *Type*: `float`  
     *Example*: `3.0`

   * **negative_guidance**  
     *Description*: Guidance of negative training used to train.  
     *Type*: `float`  
     *Example*: `1.0`

   * **ddim_eta**  
     *Description*: DDIM eta parameter for sampling.  
     *Type*: `int` or `float`  
     *Example*: `0`

4. **Training Setup**

   * **prompt**  
     *Description*: Prompt corresponding to the concept to erase.  
     *Type*: `str`  
     *Example*: `nudity`

   * **dataset_retain**  
     *Description*: Prompts corresponding to non-target concepts to retain.  
     *Type*: `str`  
     *Choices*: `coco_object`, `coco_object_no_filter`, `imagenet243`, `imagenet243_no_filter`  
     *Example*: `coco_object`

   * **retain_batch**  
     *Description*: Batch size of retaining prompts during training.  
     *Type*: `int`  
     *Example*: `5`

   * **retain_train**  
     *Description*: Retaining training mode; choose between iterative (`iter`) or regularization (`reg`).  
     *Type*: `str`  
     *Choices*: `iter`, `reg`  
     *Example*: `iter`

   * **retain_step**  
     *Description*: Number of steps for retaining prompts.  
     *Type*: `int`  
     *Example*: `1`

   * **retain_loss_w**  
     *Description*: Retaining loss weight.  
     *Type*: `float`  
     *Example*: `1.0`

   * **train_method**  
     *Description*: Method of training.  
     *Type*: `str`  
     *Choices*:  
       `text_encoder_full`, `text_encoder_layer0`, `text_encoder_layer01`, `text_encoder_layer012`, `text_encoder_layer0123`, `text_encoder_layer01234`, `text_encoder_layer012345`, `text_encoder_layer0123456`, `text_encoder_layer01234567`, `text_encoder_layer012345678`, `text_encoder_layer0123456789`, `text_encoder_layer012345678910`, `text_encoder_layer01234567891011`, `text_encoder_layer0_11`, `text_encoder_layer01_1011`, `text_encoder_layer012_91011`, `noxattn`, `selfattn`, `xattn`, `full`, `notime`, `xlayer`, `selflayer`  
     *Example*: `text_encoder_full`

   * **norm_layer**  
     *Description*: Flag indicating whether to update the norm layer during training.  
     *Type*: `bool`  
     *Example*: `False`

   * **attack_method**  
     *Description*: Method for adversarial attack training.  
     *Type*: `str`  
     *Choices*: `pgd`, `multi_pgd`, `fast_at`, `free_at`  
     *Example*: `pgd`

   * **component**  
     *Description*: Component to apply the attack on.  
     *Type*: `str`  
     *Choices*: `all`, `ffn`, `attn`  
     *Example*: `all`

   * **iterations**  
     *Description*: Total number of training iterations.  
     *Type*: `int`  
     *Example*: `10`  
     *(Note: The help argument may default to a higher value, e.g., 1000, but the config file sets it to 10.)*

   * **save_interval**  
     *Description*: Interval (in iterations) at which checkpoints are saved.  
     *Type*: `int`  
     *Example*: `200`

   * **lr**  
     *Description*: Learning rate used during training.  
     *Type*: `float`  
     *Example*: `1e-5`

5. **Adversarial Attack Hyperparameters**

   * **adv_prompt_num**  
     *Description*: Number of prompt tokens for adversarial soft prompt learning.  
     *Type*: `int`  
     *Example*: `1`

   * **attack_embd_type**  
     *Description*: The adversarial embedding type; options are word embedding or condition embedding.  
     *Type*: `str`  
     *Choices*: `word_embd`, `condition_embd`  
     *Example*: `word_embd`

   * **attack_type**  
     *Description*: The type of adversarial attack applied to the prompt.  
     *Type*: `str`  
     *Choices*: `replace_k`, `add`, `prefix_k`, `suffix_k`, `mid_k`, `insert_k`, `per_k_words`  
     *Example*: `prefix_k`

   * **attack_init**  
     *Description*: Strategy for initializing the adversarial attack; either randomly or using the latest parameters.  
     *Type*: `str`  
     *Choices*: `random`, `latest`  
     *Example*: `latest`

   * **attack_step**  
     *Description*: Number of steps for the adversarial attack.  
     *Type*: `int`  
     *Example*: `30`

   * **attack_init_embd**  
     *Description*: Initial embedding for the attack (optional).  
     *Type*: Depends on implementation; default is `None`  
     *Example*: `None`

   * **adv_prompt_update_step**  
     *Description*: Frequency (in iterations) at which the adversarial prompt is updated.  
     *Type*: `int`  
     *Example*: `1`

   * **attack_lr**  
     *Description*: Learning rate for adversarial attack training.  
     *Type*: `float`  
     *Example*: `1e-3`

   * **warmup_iter**  
     *Description*: Number of warmup iterations before starting the adversarial attack.  
     *Type*: `int`  
     *Example*: `200`

6. **Backend**

   * **backend**  
     *Description*: Backend framework to be used (e.g., CompVis).  
     *Type*: `str`  
     *Example*: `compvis`
     *Choices*: `compvis` or `diffusers`


## Directory Structure

- `algorithm.py`: Implementation of the AdvUnlearnAlgorithm class.
- `configs/`: Contains configuration files for AdvUnlearn for compvis and diffusers.
- `model.py`: Implementation of the AdvUnlearnModel class for compvis and diffusers.
- `trainer.py`: Trainer for adversarial unlearning for compvis and diffusers.
- `utils.py`: Utility functions used in the project.
- `dataset_handler.py`: handles prompt cleaning and retaining dataset creation for adversarial unlearning.
- `compvis_trainer.py`: Trainer for adversarial unlearning for compvis.
- `diffusers_trainer.py`: Trainer for adversarial unlearning for diffusers.



### Evaluation for mu_defense

This section provides instructions for running the **evaluation framework** for the unlearned Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying adversial unlearning.

#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file.  


**Example code**

**Run with default config**

```python
from mu_defense.algorithms.adv_unlearn import MUDefenseEvaluator
from mu_defense.algorithms.adv_unlearn.configs import mu_defense_evaluation_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu
from evaluation.metrics.clip import clip_score
from evaluation.metrics.fid import fid_score

target_ckpt = "outputs/results_with_retaining/nudity/coco_object/pgd/AttackLr_0.001/text_encoder_full/all/prefix_k/AdvUnlearn-nudity-method_text_encoder_full_all-Attack_pgd-Retain_coco_object_iter_1.0-lr_1e-05-AttackLr_0.001-prefix_k_adv_num_1-word_embd-attack_init_latest-attack_step_30-adv_update_1-warmup_iter_200/models/Diffusers-UNet-noxattn-epoch_0.pt"
evaluator = MUDefenseEvaluator(config=mu_defense_evaluation_config) #default config

gen_image_path = evaluator.generate_images() #generate images for evaluation
print(gen_image_path)  

prompt_path = "data/prompts/sample_prompt.csv"
ref_image_path = "coco_dataset/extracted_files/coco_sample"
device = "0"
clip_val = clip_score(gen_image_path, prompt_path, device)    
print(clip_val)    

fid_val, _  = fid_score(gen_image_path, ref_image_path)
print(fid_val)
```

**Run with your configs**

Check the config descriptions to use your own confgs.

```python
from mu_defense.algorithms.adv_unlearn import MUDefenseEvaluator
from mu_defense.algorithms.adv_unlearn.configs import mu_defense_evaluation_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu
from evaluation.metrics.clip import clip_score
from evaluation.metrics.fid import fid_score

target_ckpt = "outputs/results_with_retaining/nudity/coco_object/pgd/AttackLr_0.001/text_encoder_full/all/prefix_k/AdvUnlearn-nudity-method_text_encoder_full_all-Attack_pgd-Retain_coco_object_iter_1.0-lr_1e-05-AttackLr_0.001-prefix_k_adv_num_1-word_embd-attack_init_latest-attack_step_30-adv_update_1-warmup_iter_200/models/Diffusers-UNet-noxattn-epoch_0.pt"
evaluator = MUDefenseEvaluator(config=mu_defense_evaluation_config,
                               target_ckpt = target_ckpt,
                               model_config_path = erase_diff_train_mu.model_config_path,
                               save_path = "outputs/adv_unlearn/results",
                               prompts_path = "data/prompts/sample_prompt.csv",
                               num_samples = 1,
                               folder_suffix = "imagenette",
                               devices = "0",)

gen_image_path = evaluator.generate_images() #generates images for evaluation
print(gen_image_path)  

prompt_path = "data/prompts/sample_prompt.csv"
ref_image_path = "coco_dataset/extracted_files/coco_sample"
device = "0"
clip_val = clip_score(gen_image_path, prompt_path, device)    
print(clip_val)    

fid_val, _  = fid_score(gen_image_path, ref_image_path)
print(fid_val)

```

**Running the image generation Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```

**How It Works** 

* Default Values: The script first loads default values from the evluation config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the evaluation. 


## Description of Evaluation Configuration Parameters

- **model_name:**  
  **Type:** `str`  
  **Description:** Name of the model to use. Options include `"SD-v1-4"`, `"SD-V2"`, `"SD-V2-1"`, etc.
  **required:** False

  - **encoder_model_name_or_path**  
     *Description*: Model name or path for the encoder.
     *Type*: `str`  
     *Example*: `CompVis/stable-diffusion-v1-4`

- **target_ckpt:**  
  **Type:** `str`  
  **Description:** Path to the target checkpoint.  
  - If empty, the script will load the default model weights.  
  - If provided, it supports both Diffusers-format checkpoints (directory) and CompVis checkpoints (file ending with `.pt`). For CompVis, use the checkpoint of the model saved as Diffuser format.

- **save_path:**  
  **Type:** `str`  
  **Description:** Directory where the generated images will be saved.

- **prompts_path:**  
  **Type:** `str`  
  **Description:** Path to the CSV file containing prompts, evaluation seeds, and case numbers.  
  **Default:** `"data/prompts/visualization_example.csv"`

- **guidance_scale:**  
  **Type:** `float`  
  **Description:** Parameter that controls the classifier-free guidance during generation.  
  **Default:** `7.5`

- **image_size:**  
  **Type:** `int`  
  **Description:** Dimensions of the generated images (height and width).  
  **Default:** `512`

- **ddim_steps:**  
  **Type:** `int`  
  **Description:** Number of denoising steps (used in the diffusion process).  
  **Default:** `100`

- **num_samples:**  
  **Type:** `int`  
  **Description:** Number of samples generated for each prompt.  
  **Default:** `1`

- **from_case:**  
  **Type:** `int`  
  **Description:** Minimum case number from which to start generating images.  
  **Default:** `0`

- **folder_suffix:**  
  **Type:** `str`  
  **Description:** Suffix added to the output folder name for visualizations.

- **prompt_path:**  
  **Type:** `str`  
  **Description:** Path to the CSV file containing prompts for evaluation.  
  **Example:** `"data/prompts/coco_10k.csv"`

- **devices:**  
  **Type:** `str`  
  **Description:** Comma-separated list of device IDs to be used during evaluation.  
  **Example:** `"0,0"`
