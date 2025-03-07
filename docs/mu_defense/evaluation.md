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
