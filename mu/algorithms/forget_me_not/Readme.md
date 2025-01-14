# Forget Me Not Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The Forget Me Not algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

### Installation
```
pip install unlearn_diff
```
### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

```bash
conda --version
```
### Create environment:
```
create_env <algorithm_name>
```
eg: ```create_env forget_me_not```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate forget_me_not```

The <algorithm_name> has to be one of the folders in the `mu/algorithms` folder.

### Downloading data and models.
After you install the package, you can use the following commands to download.

1. **Dataset**:
  - **i2p**:
    - **Sample**:
     ```
     download_data sample i2p
     ```
    - **Full**:
     ```
     download_data full i2p
     ```
  - **quick_canvas**:
    - **Sample**:
     ```
     download_data sample quick_canvas
     ```
    - **Full**:
     ```
     download_data full quick_canvas
     ```

2. **Model**:
  - **compvis**:
    ```
    download_model compvis
    ```
  - **diffuser**:
    ```
    download_model diffuser
    ```

**Verify the Downloaded Files**

After downloading, verify that the datasets have been correctly extracted:
```bash
ls -lh ./data/i2p-dataset/sample/
ls -lh ./data/quick-canvas-dataset/sample/
```
---

### Example Command

1. **Train a Text Inversion**

```bash
python -m mu.algorithms.forget_me_not.scripts.train_ti \
--config_path mu/algorithms/forget_me_not/config/train_ti_config.yaml
```

**Running the Script in Offlikne Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.forget_me_not.scripts.train_ti \
--config_path mu/algorithms/forget_me_not/config/train_ti_config.yaml
```

2. **Perform Unlearning**

Before running the train_attn.py script, update the ti_weights_path parameter in the configuration file to point to the output generated from the Text Inversion (train_ti.py) stage.

Example:
`ti_weights_path: "outputs/forget_me_not/ti_models/step_inv_10.safetensors"`

Run unlearning script:

```bash
python -m mu.algorithms.forget_me_not.scripts.train_attn \
--config_path mu/algorithms/forget_me_not/config/train_attn_config.yaml
```

**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.forget_me_not.scripts.train_attn \
--config_path mu/algorithms/forget_me_not/config/train_attn_config.yaml
```

**Passing Arguments via the Command Line**

The `train_ti.py` and `train_attn.py` script allows you to override configuration parameters specified in the `train_ti_config.yaml` and `train_attn_config.yaml` files by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.

**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.forget_me_not.scripts.train_ti \
    --config_path mu/algorithms/forget_me_not/config/train_ti_config.yaml \
    --ckpt_path /path/to/style50 \
    --raw_dataset_dir /path/to/raw_dataset \
```

**Explanation of the Example**

    * --config_path: Path to the pretrained model's checkpoint file for Stable Diffusion.

    * --raw_dataset_dir: Directory containing the original dataset organized by themes and classes.


**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.


## Directory Structure

- `algorithm.py`: Implementation of the Forget Me NotAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Forget Me NotModel class.
- `scripts/train.py`: Script to train the Forget Me Not algorithm.
- `trainer.py`: Implementation of the Forget Me NotTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---

**This method involves two stages:**

1. **Train a Text Inversion**: The first stage involves training a Text Inversion. Refer to the script [`train_ti.py`](mu/algorithms/forget_me_not/scripts/train_ti.py) for details and implementation. It uses `train_ti_config.yaml` as config file.

2. **Perform Unlearning**: The second stage uses the outputs from the first stage to perform unlearning. Refer to the script [`train_attn.py`](mu/algorithms/forget_me_not/scripts/train_attn.py) for details and implementation. It uses `train_attn_config.yaml` as config file.



### Description of Arguments in train_ti_config.yaml

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Training Configuration**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **steps**: Number of training steps.
- **lr**: Learning rate for the training optimizer.
- **weight_decay_ti**: Weight decay for Text Inversion training.
- **seed**: Random seed for reproducibility.
- **placeholder_tokens**: Tokens used as placeholders during training.
- **placeholder_token_at_data**: Placeholders used in the dataset for Text Inversion training.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_batch_size**: Batch size for training.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).



### Description of Arguments in train_attn_config.yaml

### Key Parameters

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Text Inversion**

- **use_ti**: Boolean indicating whether to use Text Inversion weights.
- **ti_weights_path**: File path to the Text Inversion model weights.

**Tokens**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **placeholder_tokens**: Tokens used as placeholders during training.

**Training Configuration**

- **mixed_precision**: Precision type to use during training (e.g., `fp16` or `fp32`).
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_text_encoder**: Boolean to enable or disable training of the text encoder.
- **enable_xformers_memory_efficient_attention**: Boolean to enable memory-efficient attention mechanisms.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **allow_tf32**: Boolean to allow TensorFloat-32 computation for faster training.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **train_batch_size**: Batch size for training.
- **use_8bit_adam**: Boolean to enable or disable 8-bit Adam optimizer.
- **adam_beta1**: Beta1 parameter for the Adam optimizer.
- **adam_beta2**: Beta2 parameter for the Adam optimizer.
- **adam_weight_decay**: Weight decay for the Adam optimizer.
- **adam_epsilon**: Epsilon value for the Adam optimizer.
- **size**: Image resolution size for training.
- **with_prior_preservation**: Boolean indicating whether to use prior preservation during training.
- **num_train_epochs**: Number of training epochs.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.
- **lr_num_cycles**: Number of cycles for learning rate scheduling.
- **lr_power**: Exponent to control the shape of the learning rate curve.
- **max-steps**: Maximum number of training steps.
- **no_real_image**: Boolean to skip using real images in training.
- **max_grad_norm**: Maximum norm for gradient clipping.
- **checkpointing_steps**: Number of steps between model checkpoints.
- **set_grads_to_none**: Boolean to set gradients to None instead of zeroing them out.
- **lr**: Learning rate for the training optimizer.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).

**Miscellaneous**

- **only-xa**: Boolean to enable additional configurations specific to the XA pipeline.



#### forget_me_not Evaluation Framework

This section provides instructions for running the **evaluation framework** for the forget_me_not algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/forget_me_not/scripts/` directory.

### **Basic Command to Run Evaluation:**

```bash
conda activate <env_name>
```

```bash
python -m mu.algorithms.forget_me_not.scripts.evaluate \
--config_path mu/algorithms/forget_me_not/configs/evaluation_config.yaml
```


**Running in Offline Mode:**

```bash
WANDB_MODE=offline python -m mu.algorithms.forget_me_not.scripts.evaluate \
--config_path mu/algorithms/forget_me_not/configs/evaluation_config.yaml
```


**Example with CLI Overrides:**

```bash
python -m mu.algorithms.forget_me_not.scripts.evaluate \
    --config_path mu/algorithms/forget_me_not/configs/evaluation_config.yaml \
    --devices "0" \
    --seed 123 \
    --cfg_text 8.5 \
    --batch_size 16
```


#### **Description of parameters in evaluation_config.yaml**

The `evaluation_config.yaml` file contains the necessary parameters for running the forget_me_not evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/forget_me_not/finetuned_models/forget_me_not_Abstractionism_model.pth"`

- classification_model : Specifies the classification model used for evaluating the generated outputs.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`

- model_ckpt_path: Path to pretrained Stable Diffusion model.
   - *Type*: `str`
   - *Example*: `models/diffuser/style50`

---

### **Training and Sampling Parameters:**
- theme : Specifies the theme or concept being evaluated for removal from the model's outputs.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`

- devices : CUDA device IDs to be used for the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"0"`  

- cfg_text_list : Classifier-free guidance scale value for image generation. Higher values increase the strength of the conditioning prompt.  
   - *Type:* `list`  
   - *Example:* `[9.0]`  

- seed : Random seed for reproducibility of results.  
   - *Type:* `int`  
   - *Example:* `188`

- ddim_steps : Number of steps for the DDIM (Denoising Diffusion Implicit Models) sampling process.  
   - *Type:* `int`  
   - *Example:* `100`

- ddim_eta : DDIM eta value for controlling the amount of randomness during sampling. Set to `0` for deterministic sampling.  
   - *Type:* `float`  
   - *Example:* `0.0`

- image_height : Height of the generated images in pixels.  
   - *Type:* `int`  
   - *Example:* `512`

- image_width : Width of the generated images in pixels.  
   - *Type:* `int`  
   - *Example:* `512`

---

### **Output and Logging Parameters:**
- sampler_output_dir : Directory where generated images will be saved during evaluation.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/forget_me_not/"`

- eval_output_dir : Directory where evaluation metrics and results will be stored.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/forget_me_not/"`

- reference_dir : Directory containing original images for comparison during evaluation.  
   - *Type:* `str`  
   - *Example:* `"/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/"`

---

### **Performance and Efficiency Parameters:**
- multiprocessing : Enables multiprocessing for faster evaluation for FID score. Recommended for large datasets.  
   - *Type:* `bool`  
   - *Example:* `False`  

- batch_size : Batch size used during FID computation and evaluation.  
   - *Type:* `int`  
   - *Example:* `16`  

---

### **Optimization Parameters:**
- forget_theme : Concept or style intended for removal in the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`

- seed_list : List of random seeds for performing multiple evaluations with different randomness levels.  
   - *Type:* `list`  
   - *Example:* `["188"]`





