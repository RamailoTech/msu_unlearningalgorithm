
# Concept Ablation Algorithm for Machine Unlearning

This repository provides an implementation of the Concept Ablation algorithm for machine unlearning in Stable Diffusion models. The Concept Ablation algorithm enables the removal of specific concepts or styles from a pre-trained model without the need for retraining from scratch.

---

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/concept_ablation/environment.yaml -n mu_concept_ablation
```

```bash
conda --version
```
### Create environment:
```
create_env <algorithm_name>
```
eg: ```create_env concept_ablation```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate concept_ablation```

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

## Run Train

### Example Command

```bash
python -m mu.algorithms.concept_ablation.scripts.train \
--config_path mu/algorithms/concept_ablation/configs/train_config.yaml \
--prompts mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt
```

### Running the Training Script in Offline Mode

```bash
WANDB_MODE=offline python -m mu.algorithms.concept_ablation.scripts.train \
--config_path mu/algorithms/concept_ablation/configs/train_config.yaml \
--prompts /home/ubuntu/Projects/Palistha/msu_unlearningalgorithm/mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt
```

### Overriding Configuration via Command Line

You can override configuration parameters by passing them directly as arguments during runtime.

**Example Usage with Command-Line Arguments:**

```bash
python -m mu.algorithms.concept_ablation.scripts.train \
--config_path mu/algorithms/concept_ablation/configs/train_config.yaml \
--batch_size 8 \
--base_lr 1e-5 \
--devices 0,1 \
--output_dir outputs/experiment_2
```

**Explanation:**
* `--config_path`: Specifies the YAML configuration file.
* `--batch_size`: Overrides the batch size to 8.
* `--base_lr`: Updates the base learning rate to 1e-5.
* `--devices`: Specifies the GPUs (e.g., device 0 and 1).
* `--output_dir`: Sets a custom output directory for the experiment.

---

## Directory Structure

- `algorithm.py`: Core implementation of the Concept Ablation Algorithm.
- `configs/`: Configuration files for training and generation.
- `data_handler.py`: Data handling and preprocessing.
- `scripts/train.py`: Script to train the Concept Ablation Algorithm.
- `callbacks/`: Custom callbacks for logging and monitoring training.
- `utils.py`: Utility functions.

---

## How It Works

1. **Default Configuration:** Loads values from the specified YAML file (`--config_path`).
2. **Command-Line Overrides:** Updates the configuration with values provided as command-line arguments.
3. **Training Execution:** Initializes the `ConceptAblationAlgorithm` and trains the model using the provided dataset, model checkpoint, and configuration.
4. **Output:** Saves the fine-tuned model and logs training metrics in the specified output directory.

---

## Notes

1. Ensure all dependencies are installed as per the environment file.
2. The training process generates logs in the `logs/` directory for easy monitoring.
3. Use appropriate CUDA devices for optimal performance during training.
4. Regularly verify dataset and model configurations to avoid errors during execution.


## Configuration File (`train_config.yaml`)

### Training Parameters

* **seed:** Random seed for reproducibility.
    * Type: int
    * Example: 23

* **scale_lr:** Whether to scale the base learning rate.
    * Type: bool
    * Example: True

* **caption_target:** Target style to remove.
    * Type: str
    * Example: "Abstractionism Style"

* **regularization:** Adds regularization loss during training.
    * Type: bool
    * Example: True

* **n_samples:** Number of batch sizes for image generation.
    * Type: int
    * Example: 10

* **train_size:** Number of generated images for training.
    * Type: int
    * Example: 1000

* **base_lr:** Learning rate for the optimizer.
    * Type: float
    * Example: 2.0e-06

### Model Configuration

* **model_config_path:** Path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/model_config.yaml"

* **ckpt_path:** Path to the Stable Diffusion model checkpoint.
    * Type: str
    * Example: "/path/to/compvis.ckpt"

### Dataset Directories

* **raw_dataset_dir:** Directory containing the raw dataset categorized by themes or classes.
    * Type: str
    * Example: "/path/to/raw_dataset"

* **processed_dataset_dir:** Directory to save the processed dataset.
    * Type: str
    * Example: "/path/to/processed_dataset"

* **dataset_type:** Specifies the dataset type for training.
    * Choices: ["unlearncanvas", "i2p"]
    * Example: "unlearncanvas"

* **template:** Type of template to use during training.
    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* **template_name:** Name of the concept or style to erase.
    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"

### Output Configurations

* **output_dir:** Directory to save fine-tuned models and results.
    * Type: str
    * Example: "outputs/concept_ablation/finetuned_models"

### Device Configuration

* **devices:** CUDA devices for training (comma-separated).
    * Type: str
    * Example: "0"


#### Concept ablation Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Concept ablation algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/concept_ablation/scripts/` directory.

### **Basic Command to Run Evaluation:**

```bash
conda activate <env_name>
```

```bash
python -m mu.algorithms.concept_ablation.scripts.evaluate \
--config_path mu/algorithms/concept_ablation/configs/evaluation_config.yaml
```


**Running in Offline Mode:**

```bash
WANDB_MODE=offline python -m mu.algorithms.concept_ablation.scripts.evaluate \
--config_path mu/algorithms/concept_ablation/configs/evaluation_config.yaml
```


**Example with CLI Overrides:**

```bash
python -m mu.algorithms.concept_ablation.scripts.evaluate \
    --config_path mu/algorithms/concept_ablation/configs/evaluation_config.yaml \
    --devices "0" \
    --seed 123 \
    --cfg_text 8.5 \
    --batch_size 16
```


#### **Description of parameters in evaluation_config.yaml**

The `evaluation_config.yaml` file contains the necessary parameters for running the Concept ablation evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- model_config : Path to the YAML file specifying the model architecture and settings.  
   - *Type:* `str`  
   - *Example:* `"mu/algorithms/concept_ablation/configs/model_config.yaml"`

- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/concept_ablation/finetuned_models/concept_ablation_Abstractionism_model.pth"`

- classification_model : Specifies the classification model used for evaluating the generated outputs.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`

- model_ckpt_path: Path to pretrained Stable Diffusion model.
   - *Type*: `str`
   - *Example*: `models/compvis/style50/compvis.ckpt`

---

### **Training and Sampling Parameters:**
- theme : Specifies the theme or concept being evaluated for removal from the model's outputs.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`

- devices : CUDA device IDs to be used for the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"0"`  

- cfg_text : Classifier-free guidance scale value for image generation. Higher values increase the strength of the conditioning prompt.  
   - *Type:* `float`  
   - *Example:* `9.0`  

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
   - *Example:* `"outputs/eval_results/mu_results/concept_ablation/"`

- eval_output_dir : Directory where evaluation metrics and results will be stored.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/concept_ablation/"`

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