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
