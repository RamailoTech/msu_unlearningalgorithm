# Semi Permeable Membrane Algorithm for Machine Unlearning

This repository provides an implementation of the semipermeable membrane algorithm for machine unlearning in Stable Diffusion models. The semipermeable membrane algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.


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
eg: ```create_env semipermeable_membrane```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate semipermeable_membrane```

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
## Usage

To train the Semi Permeable Membrane algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m mu.algorithms.semipermeable_membrane.scripts.train \
--config_path mu/algorithms/semipermeable_membrane/config/train_config.yaml
```

**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.semipermeable_membrane.scripts.train \
--config_path mu/algorithms/semipermeable_membrane/config/train_config.yaml
```



**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.



**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.semipermeable_membrane.scripts.train \
--config_path mu/algorithms/semipermeable_membrane/config/train_config.yaml \
--dataset_type unlearncanvas \
--template object \
--template_name self-harm \
--devices 0,1 \
--output_dir outputs/experiment_1 \
--use_sample
```


**Explanation of the Example**

* --config_path: Specifies the YAML configuration file to load default values.

* --dataset_type: Defines the dataset type (e.g., unlearncanvas or i2p).

* --template: Specifies the template to use (e.g., object, style, or i2p).

* --template_name: The specific name for the template being used (e.g., self-harm, Abstractionism).

* --devices: Comma-separated list of CUDA devices to use for training (e.g., 0,1 for using GPU 0 and GPU 1).

* --output_dir: Sets a custom output directory for the results of this run.

* --use_sample: Specifies whether to use a sample dataset for training.


**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.

## Directory Structure

- `algorithm.py`: Implementation of the Semi Permeable MembraneAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Semi Permeable MembraneModel class.
- `scripts/train.py`: Script to train the Semi Permeable Membrane algorithm.
- `trainer.py`: Implementation of the Semi Permeable MembraneTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---
### Description of Arguments in train_config.yaml

**pretrained_model**

* ckpt_path: File path to the pretrained model's checkpoint file.

* v2: Boolean indicating whether the pretrained model is version 2 or not.

* v_pred: Boolean to enable/disable "v-prediction" mode for diffusion models.

* clip_skip: Number of CLIP layers to skip during inference.

**network**

* rank: Rank of the low-rank adaptation network.

* alpha: Scaling factor for the network during training.


**train**

* precision: Numerical precision to use during training (e.g., float32 or float16).

* noise_scheduler: Type of noise scheduler to use in the training loop (e.g., ddim).

* iterations: Number of training iterations.

* batch_size: Batch size for training.

* lr: Learning rate for the training optimizer.

* unet_lr: Learning rate for the U-Net model.

* text_encoder_lr: Learning rate for the text encoder.

* optimizer_type: Optimizer to use for training (e.g., AdamW8bit).

* lr_scheduler: Learning rate scheduler to apply during training.

* lr_warmup_steps: Number of steps for linear warmup of the learning rate.

* lr_scheduler_num_cycles: Number of cycles for a cosine-with-restarts scheduler.

* max_denoising_steps: Maximum denoising steps to use during training.

**save**

* per_steps: Frequency of saving the model (in steps).

* precision: Numerical precision for saved model weights


**other**

* use_xformers: Boolean to enable xformers memory-efficient attention.

* wandb_project and wandb_run

* Configuration for tracking the training progress using Weights & Biases.

* wandb_project: Project name in W&B.

* wandb_run: Specific run name in the W&B dashboard.

**use_sample**

* Boolean to indicate whether to use the sample dataset for training.

**dataset_type**

* Type of dataset to use, options are unlearncanvas or i2p.

**template**
* Specifies the template type, choices are:
    * object: Focus on specific objects.
    * style: Focus on artistic styles.
    * i2p: Intermediate style processing.

**template_name**

* Name of the template, choices are:
    * self-harm
    * Abstractionism

**prompt**

* target: Target template or concept to guide training (references template_name).

* positive: Positive prompt based on the template.

* unconditional: Unconditional prompt text.

* neutral: Neutral prompt text.

* action: Specifies the action applied to the prompt (e.g., erase_with_la).

* guidance_scale: Guidance scale for classifier-free guidance.

* resolution: Image resolution for training.

* batch_size: Batch size for generating prompts.

* dynamic_resolution: Boolean to allow dynamic resolution.

* la_strength: Strength of local adaptation.

* sampling_batch_size: Batch size for sampling images.

**devices**

* CUDA devices to use for training (specified as a comma-separated list, e.g., "0,1").

**output_dir**

* Directory to save the fine-tuned model and other outputs.

**verbose**

* Boolean flag for verbose logging during training.


#### Semipermeable membrane Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Semipermeable membrane algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/semipermeable_membrane/scripts/` directory.

### **Basic Command to Run Evaluation:**

```bash
conda activate <env_name>
```

```bash
python -m mu.algorithms.semipermeable_membrane.scripts.evaluate \
--config_path mu/algorithms/semipermeable_membrane/configs/evaluation_config.yaml
```


**Running in Offline Mode:**

```bash
WANDB_MODE=offline python -m mu.algorithms.semipermeable_membrane.scripts.evaluate \
--config_path mu/algorithms/semipermeable_membrane/configs/evaluation_config.yaml
```


**Example with CLI Overrides:**

```bash
python -m mu.algorithms.semipermeable_membrane.scripts.evaluate \
    --config_path mu/algorithms/semipermeable_membrane/configs/evaluation_config.yaml \
    --devices "0" \
    --seed 123 \
    --cfg_text 8.5 \
    --batch_size 16
```


#### **Description of parameters in evaluation_config.yaml**

The `evaluation_config.yaml` file contains the necessary parameters for running the Semipermeable membrane evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration Parameters:**
- spm_path: paths to finetuned model checkpoint.
   - *Type:* `list`
   - *Example:* `outputs/semipermeable_membrane/finetuned_models/semipermeable_membrane_Abstractionism_last.safetensors`

- base_model : Path to the pre-trained base model used for image generation.  
   - *Type:* `str`  
   - *Example:* `"path/to/base/model.pth"`  

- precision : Specifies the numerical precision for model computation.  
   - *Type:* `str`  
   - *Options:* `"fp32"`, `"fp16"`, `"bf16"`  
   - *Example:* `"fp32"`  

- spm_multiplier:  Specifies the multiplier for Semipermeable Membrane (SPM) model.  
   - *Type:* `float`  
   - *Example:* `1.0`  

- v2 : Specifies whether to use version 2.x of the model.  
   - *Type:* `bool`  
   - *Example:* `false`  

- matching_metric : Metric used for evaluating the similarity between generated prompts and erased concepts.  
   - *Type:* `str`  
   - *Options:* `"clipcos"`, `"clipcos_tokenuni"`, `"tokenuni"`  
   - *Example:* `"clipcos_tokenuni"`  

- model_config : Path to the model configuration YAML file.  
   - *Type:* `str`  
   - *Example:* `"mu/algorithms/semipermeable_membrane/config"`  

- model_ckpt_path: Path to pretrained Stable Diffusion model.
   - *Type*: `str`
   - *Example*: `models/diffuser/style50`

---

### **Sampling Parameters:**
- theme : Specifies the theme for the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`  

- seed : Random seed for reproducibility of the evaluation process.  
   - *Type:* `int`  
   - *Example:* `188`  

- devices : Specifies the CUDA devices for running the model.  
   - *Type:* `str` (Comma-separated for multiple devices)  
   - *Example:* `"0"`  

- task : Specifies the task type for the evaluation process.  
   - *Type:* `str`  
   - *Options:* `"class"`, `"style"`  
   - *Example:* `"class"`  

---

### **Output Parameters:**
- sampler_output_dir : Directory where generated images will be saved during the sampling process.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/semipermeable_membrane/"`  

- eval_output_dir : Directory where evaluation metrics and results will be stored.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/semipermeable_membrane/"`  


---

### **Dataset and Classification Parameters:**
- reference_dir : Path to the reference dataset used for evaluation and comparison.  
   - *Type:* `str`  
   - *Example:* `"/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/"`  

- classification_model : Specifies the classification model used for the evaluation.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`  

- forget_theme : Specifies the theme to be forgotten during the unlearning process.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`  

---

### **Performance Parameters:**
- multiprocessing : Enables multiprocessing for faster evaluation.  
   - *Type:* `bool`  
   - *Example:* `false`  

- seed_list :  List of random seeds for multiple evaluation trials.  
   - *Type:* `list`  
   - *Example:* `["188"]`  