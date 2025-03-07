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


