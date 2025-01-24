# ScissorHands Algorithm for Machine Unlearning

This repository provides an implementation of the scissor hands algorithm for machine unlearning in Stable Diffusion models. The scissor hands algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

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
eg: ```create_env scissorhands```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate scissorhands```

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

To train the ScissorHands algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**
```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import (
    scissorhands_train_mu,
)

algorithm = ScissorHandsAlgorithm(
    scissorhands_train_mu,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
)
algorithm.run()
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python my_trainer.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

### Directory Structure

- `algorithm.py`: Implementation of the ScissorHandsAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the ScissorHandsModel class.
- `scripts/train.py`: Script to train the ScissorHands algorithm.
- `trainer.py`: Implementation of the ScissorHandsTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---
### Description of Arguments in train_config.yaml

**Training Parameters**

* train_method: Specifies the method of training for concept erasure.

    * Choices: ["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"]
    * Example: "xattn"

* alpha: Guidance strength for the starting image during training.

    * Type: float
    * Example: 0.1

* epochs: Number of epochs to train the model.

    * Type: int
    * Example: 1


**Model Configuration**

* model_config_path: File path to the Stable Diffusion model configuration YAML file.

    * type: str
    * Example: "/path/to/model_config.yaml"

* ckpt_path: File path to the checkpoint of the Stable Diffusion model.

    * Type: str
    * Example: "/path/to/model_checkpoint.ckpt"


**Dataset Directories**

* raw_dataset_dir: Directory containing the raw dataset categorized by themes or classes.

    * Type: str
    * Example: "/path/to/raw_dataset"

* processed_dataset_dir: Directory to save the processed dataset.

    * Type: str
    * Example: "/path/to/processed_dataset"

* dataset_type: Specifies the dataset type for the training process.

    * Choices: ["unlearncanvas", "i2p"]
    * Example: "unlearncanvas"

* template: Type of template to use during training.

    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* template_name: Name of the specific concept or style to be erased.

    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"


**Output Configurations**

* output_dir: Directory where the fine-tuned models and results will be saved.

    * Type: str
    * Example: "outputs/erase_diff/finetuned_models"

**Sampling and Image Configurations**

* sparsity: Threshold for mask.

    * Type: float
    * Example: 0.99

* project: 
    * Type: bool
    * Example: false

* memory_num: 
    * Type: Int
    * Example: 1

* prune_num: 
    * Type: Int
    * Example: 1

**Device Configuration**

* devices: Specifies the CUDA devices to be used for training (comma-separated).

    * Type: str (Comma separated)
    * Example: "0, 1"


**Additional Flags**

* use_sample: Flag to indicate whether a sample dataset should be used for training.

    * Type: bool
    * Example: True


#### Scissorshands Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Scissorshands algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/scissorshands/scripts/` directory.

#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file.  

**Example Code**

```python
from mu.algorithms.scissorhands import ScissorHandsEvaluator
from mu.algorithms.scissorhands.configs import (
    scissorhands_evaluation_config
)

evaluator = ScissorHandsEvaluator(
    scissorhands_evaluation_config,
    ckpt_path="/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
    classifier_ckpt_path = "/home/ubuntu/Projects/models/classifier_ckpt_path/style50_cls.pth",
    reference_dir= "/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/"
)
evaluator.run()
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```

**How It Works** 
* Default Values: The script first loads default values from the evluation config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the evaluation. 


#### **Description of parameters in evaluation_config**

The `evaluation_config` contains the necessary parameters for running the Scissorshands evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- model_config : Path to the YAML file specifying the model architecture and settings.  
   - *Type:* `str`  
   - *Example:* `"mu/algorithms/scissorshands/configs/model_config.yaml"`

- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/scissorshands/finetuned_models/scissorshands_Abstractionism_model.pth"`

- classification_model : Specifies the classification model used for evaluating the generated outputs.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`

- classifier_ckpt_path: Path to classifer checkpoint.
   - *Type*: `str`
   - *Example*: `models/classifier_ckpt_path/style50_cls.pth`

---

### **Training and Sampling Parameters:**

- forget_theme : Concept or style intended for removal in the evaluation process.  
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
   - *Example:* `"outputs/eval_results/mu_results/scissorshands/"`

- eval_output_dir : Directory where evaluation metrics and results will be stored.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/scissorshands/"`

- reference_dir : Directory containing original images for comparison during evaluation.  
   - *Type:* `str`  
   - *Example:* `"data/quick-canvas-dataset/sample/"`


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

- seed_list : List of random seeds for performing multiple evaluations with different randomness levels.  
   - *Type:* `list`  
   - *Example:* `["188"]`