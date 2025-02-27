### Evaluation for mu_defense

This section provides instructions for running the **evaluation framework** for the unlearned Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying adversial unlearning.


#### **Image Generator**

Before proceeding with evaluation, you must generate images using the output from mu_defense. Work within the same enviornment used for defense to perform evalaution.

To generate images add the following code snippet in `image_generator.py` and modify your configs to run the file.  


**Example code**

**Run with default config**

```python
from mu_defense.algorithms.adv_unlearn.configs import example_image_generator_config
from mu_defense.algorithms.adv_unlearn import ImageGenerator
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

def generate_image():
    generate_image = ImageGenerator(
        config = example_image_generator_config
    )
    generate_image.generate_images()

if __name__ == "__main__":
    generate_image()

```

**Run with your configs**

Check the config descriptions to use your own confgs.

```python
from mu_defense.algorithms.adv_unlearn.configs import example_image_generator_config
from mu_defense.algorithms.adv_unlearn import ImageGenerator
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

def generate_image():
    generate_image = ImageGenerator(
        config = example_image_generator_config,
        target_ckpt = "outputs/adv_unlearn/models/TextEncoder-text_encoder_full-epoch_0.pt",
        model_config_path = erase_diff_train_mu.model_config_path,
        save_path = "outputs/adv_unlearn/models",
        prompts_path = "data/prompts/sample_prompt.csv",
        num_samples = 1,
        folder_suffix = "imagenette",
        devices = "0"

    )
    generate_image.generate_images()

if __name__ == "__main__":
    generate_image()

```

**Running the image generation Script in Offline Mode**

```bash
WANDB_MODE=offline python image_generator.py
```

**How It Works** 

* Default Values: The script first loads default values from the evluation config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the evaluation. 


#### **Description of parameters in image_generator_config**


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

- **device:**  
  **Type:** `str`  
  **Description:** Device(s) used for image generation. For example, `"0"` will use `cuda:0`.

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

- **origin_or_target:**  
  **Type:** `str`  
  **Description:** Indicates whether to generate images for the `"target"` model or the `"origin"`.  
  **Default:** `"target"`



#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file.  

**Example code**


**Run with default config**

```python

from mu_defense.algorithms.adv_unlearn import MUDefenseEvaluator
from mu_defense.algorithms.adv_unlearn.configs import mu_defense_evaluation_config

def mu_defense_evaluator():
    evaluator = MUDefenseEvaluator(
        config = mu_defense_evaluation_config
    )
    evaluator.run()

if __name__ == "__main__":
    mu_defense_evaluator()
```


**Run with your own config**

```python

from mu_defense.algorithms.adv_unlearn import MUDefenseEvaluator
from mu_defense.algorithms.adv_unlearn.configs import mu_defense_evaluation_config

def mu_defense_evaluator():
    evaluator = MUDefenseEvaluator(
        config = mu_defense_evaluation_config,
        gen_imgs_path = "outputs/adv_unlearn/models_visualizations_imagenette/SD-v1-4/",
        coco_imgs_path = "coco_dataset/extracted_files/coco_sample",
        output_path = "outputs/adv_unlearn/evaluation/",
        job = "clip", #donot use this if you want to calculate both clip and fid score
    )
    evaluator.run()


if __name__ == "__main__":
    mu_defense_evaluator()
```


**Running the evaluation Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```

## Description of Evaluation Configuration Parameters

- **job:**  
  **Type:** `str`  
  **Description:** Evaluation tasks to perform. If nothing is passed it cacluates both.
  **Example:** `"fid"` or `"clip"`

- **gen_imgs_path:**  
  **Type:** `str`  
  **Description:** Path to the directory containing the generated images (from adversarial unlearning).  
  **Example:** `"outputs/adv_unlearn/models_visualizations_imagenette/SD-v1-4/"`

- **coco_imgs_path:**  
  **Type:** `str`  
  **Description:** Path to the directory containing COCO dataset images for evaluation.  
  **Example:** `"coco_dataset/extracted_files/coco_sample"`

- **prompt_path:**  
  **Type:** `str`  
  **Description:** Path to the CSV file containing prompts for evaluation.  
  **Example:** `"data/prompts/coco_10k.csv"`

- **classify_prompt_path:**  
  **Type:** `str`  
  **Description:** Path to the CSV file containing classification prompts.  
  **Example:** `"data/prompts/imagenette_5k.csv"`

- **devices:**  
  **Type:** `str`  
  **Description:** Comma-separated list of device IDs to be used during evaluation.  
  **Example:** `"0,0"`

- **classification_model_path:**  
  **Type:** `str`  
  **Description:** Path or identifier of the classification model to use.  
  **Example:** `"openai/clip-vit-base-patch32"`

- **output_path:**  
  **Type:** `str`  
  **Description:** Directory where the evaluation results will be saved.  
  **Example:** `"outputs/adv_unlearn/evaluation"`