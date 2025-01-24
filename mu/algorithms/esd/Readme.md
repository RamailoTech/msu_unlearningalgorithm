# ESD Algorithm for Machine Unlearning

This repository provides an implementation of the ESD algorithm for machine unlearning in Stable Diffusion models. The ESD algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.


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
eg: ```create_env esd```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate esd```

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
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    esd_train_mu,
)

algorithm = ESDAlgorithm(
    esd_train_mu,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ),
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

## Directory Structure

- `algorithm.py`: Implementation of the ESDAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `constants/const.py`: Constants used throughout the project.
- `model.py`: Implementation of the ESDModel class.
- `scripts/train.py`: Script to train the ESD algorithm.
- `trainer.py`: Implementation of the ESDTrainer class.
- `utils.py`: Utility functions used in the project.
---

### Description of arguments being used in train_config.yaml

The `config/train_config.yaml` file is a configuration file for training a Stable Diffusion model using the ESD (Erase Stable Diffusion) method. It defines various parameters related to training, model setup, dataset handling, and output configuration. Below is a detailed description of each section and parameter:

**Training Parameters**

These parameters control the fine-tuning process, including the method of training, guidance scales, learning rate, and iteration settings.

* train_method: Specifies the method of training to decide which parts of the model to update.

    * Type: str
    * Choices: noxattn, selfattn, xattn, full, notime, xlayer, selflayer
    * Example: xattn

* start_guidance: Guidance scale for generating initial images during training. Affects the diversity of the training set.

    * Type: float
    * Example: 0.1

* negative_guidance: Guidance scale for erasing the target concept during training.

    * Type: float
    * Example: 0.0

* iterations: Number of training iterations (similar to epochs).

    * Type: int
    * Example: 1

* lr: Learning rate used by the optimizer for fine-tuning.

    * Type: float
    * Example: 5e-5

* image_size: Size of images used during training and sampling (in pixels).

    * Type: int
    * Example: 512

* ddim_steps: Number of diffusion steps used in the DDIM sampling process.

    * Type: int
    * Example: 50


**Model Configuration**

These parameters specify the Stable Diffusion model checkpoint and configuration file.

* model_config_path: Path to the YAML file defining the model architecture and parameters.

    * Type: str
    * Example: mu/algorithms/esd/configs/model_config.yaml

* ckpt_path: Path to the finetuned Stable Diffusion model checkpoint.

    * Type: str
    * Example: '../models/compvis/style50/compvis.ckpt'


**Dataset Configuration**

These parameters define the dataset type and template for training, specifying whether to focus on objects, styles, or inappropriate content.

* dataset_type: Type of dataset used for training.

    * Type: str
    * Choices: unlearncanvas, i2p
    * Example: unlearncanvas

* template: Type of concept or style to erase during training.

    * Type: str
    * Choices: object, style, i2p
    * Example: style

* template_name: Specific name of the object or style to erase (e.g., "Abstractionism").

    * Type: str
    * Example Choices: Abstractionism, self-harm
    * Example: Abstractionism


**Output Configuration**

These parameters control where the outputs of the training process, such as fine-tuned models, are stored.

* output_dir: Directory where the fine-tuned model and training results will be saved.

    * Type: str
    * Example: outputs/esd/finetuned_models

* separator: Separator character used to handle multiple prompts during training. If set to null, no special handling occurs.

    * Type: str or null
    * Example: null

**Device Configuration**

These parameters define the compute resources for training.

* devices: Specifies the CUDA devices used for training. Provide a comma-separated list of device IDs.

    * Type: str
    * Example: 0,1

* use_sample: Boolean flag indicating whether to use a sample dataset for testing or debugging.

    * Type: bool
    * Example: True




#### ESD Evaluation Framework

This section provides instructions for running the **evaluation framework** for the ESD algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file.  

**Example Code**

```python
from mu.algorithms.esd import ESDEvaluator
from mu.algorithms.esd.configs import (
    esd_evaluation_config
)

evaluator = ESDEvaluator(
    esd_evaluation_config,
    ckpt_path="/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/esd/esd_Abstractionism_model.pth",
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

The `evaluation_config` file contains the necessary parameters for running the ESD evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- model_config : Path to the YAML file specifying the model architecture and settings.  
   - *Type:* `str`  
   - *Example:* `"mu/algorithms/esd/configs/model_config.yaml"`

- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/esd/finetuned_models/esd_Abstractionism_model.pth"`

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
   - *Example:* `"outputs/eval_results/mu_results/esd/"`

- eval_output_dir : Directory where evaluation metrics and results will be stored.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/esd/"`

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

- seed_list : List of random seeds for performing multiple evaluations with different randomness levels.  
   - *Type:* `list`  
   - *Example:* `["188"]`