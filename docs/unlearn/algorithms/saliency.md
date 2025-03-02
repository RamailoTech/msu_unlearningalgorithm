# Saliency Unlearning Algorithm for Machine Unlearning

This repository provides an implementation of the Saliency Unlearning algorithm for machine unlearning in Stable Diffusion models. The Saliency Unlearning algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

```bash
conda --version
```

**Step-by-Step Setup:**

Step 1. Create a Conda Environment Create a new Conda environment named myenv with Python 3.8.5:

```bash
conda create -n myenv python=3.8.5
```

Step 2. Activate the Environment Activate the environment to work within it:

```bash
conda activate myenv
```

Step 3. Install Core Dependencies Install PyTorch, torchvision, CUDA Toolkit, and ONNX Runtime with specific versions:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 onnxruntime==1.16.3 -c pytorch -c conda-forge
```

Step 4. Install our unlearn_diff Package using pip:

```bash
pip install unlearn_diff
```

Step 5. Install Additional Git Dependencies:

 After installing unlearn_diff, install the following Git-based dependencies in the same Conda environment to ensure full functionality:

```bash
pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```

```bash
pip install git+https://github.com/openai/CLIP.git@main#egg=clip
```

```bash
pip install git+https://github.com/crowsonkb/k-diffusion.git
```

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

```bash
pip install git+https://github.com/Phoveran/fastargs.git@main#egg=fastargs
```

```bash
pip install git+https://github.com/boomb0om/text2image-benchmark
```


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

Before training saliency unlearning algorithm you need to generate mask. Use the following code snippet to generate mask.

**Step 1: Generate mask**

**using unlearn canvas dataset**

```python
from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_generate_mask_mu

generate_mask = MaskingAlgorithm(
    saliency_unlearning_generate_mask_mu,
    ckpt_path = "models/compvis/style50/compvis.ckpt",
    raw_dataset_dir = "data/quick-canvas-dataset/sample",
    dataset_type = "unlearncanvas",
    use_sample = True, #to use sample dataset
    output_dir =  "outputs/saliency_unlearning/masks", #output path to save mask
    template_name = "Abstractionism",
    template = "style"
    )

if __name__ == "__main__":
    generate_mask.run()
```


**using i2p dataset**

```python
from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_generate_mask_i2p

generate_mask = MaskingAlgorithm(
    saliency_unlearning_generate_mask_i2p,
    ckpt_path = "models/compvis/style50/compvis.ckpt",
    raw_dataset_dir = "data/quick-canvas-dataset/sample",
    dataset_type = "unlearncanvas",
    use_sample = True, #to use sample dataset
    output_dir =  "outputs/saliency_unlearning/masks", #output path to save mask
    template_name = "self-harm",
    template = "i2p"
    )

if __name__ == "__main__":
    generate_mask.run()
```


### Run Train

**Using  quick canvas dataset**

To train the saliency unlearning algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

**Example Code**
```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearningAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_mu,
)

algorithm = SaliencyUnlearningAlgorithm(
    saliency_unlearning_train_mu,
    output_dir="/opt/dlami/nvme/outputs",
    dataset_type = "unlearncanvas"
    template_name = "Abstractionism", #concept to erase
    template = "style",
    use_sample = True #to run on sample dataset.
)
algorithm.run()
```

**Using i2p dataset**

```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearningAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_i2p,
)

algorithm = SaliencyUnlearningAlgorithm(
    saliency_unlearning_train_i2p,
    raw_dataset_dir = "data/i2p-dataset/sample",
    output_dir="/opt/dlami/nvme/outputs",
    template_name = "self-harm", #concept to erase
    template = "style",
    dataset_type = "i2p",
    use_sample = True #to run on sample dataset.
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


**Similarly, you can pass arguments during runtime to generate mask.**

**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.


### Directory Structure

- `algorithm.py`: Implementation of the SaliencyUnlearnAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the SaliencyUnlearnModel class.
- `scripts/train.py`: Script to train the SaliencyUnlearn algorithm.
- `trainer.py`: Implementation of the SaliencyUnlearnTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---
<br>

**The unlearning has two stages:**

1. Generate the mask 

2. Unlearn the weights.

<br>

### Description of configs used to generate mask:


**Model Configuration**

These parameters specify settings for the Stable Diffusion model and guidance configurations.

* c_guidance: Guidance scale used during loss computation in the model. Higher values may emphasize certain features in mask generation.
    
    * Type: float
    * Example: 7.5

* batch_size: Number of images processed in a single batch.

    * Type: int
    * Example: 4

* ckpt_path: Path to the model checkpoint file for Stable Diffusion.

    * Type: str
    * Example: /path/to/compvis.ckpt

* model_config_path: Path to the model configuration YAML file for Stable Diffusion.

    * Type: str
    * Example: /path/to/model_config.yaml

* num_timesteps: Number of timesteps used in the diffusion process.

    * Type: int
    * Example: 1000

* image_size: Size of the input images used for training and mask generation (in pixels).

    * Type: int
    * Example: 512


**Dataset Configuration**

These parameters define the dataset paths and settings for mask generation.

* raw_dataset_dir: Path to the directory containing the original dataset, organized by themes and classes.

    * Type: str
    * Example: /path/to/raw/dataset

* processed_dataset_dir: Path to the directory where processed datasets will be saved after mask generation.

    * Type: str
    * Example: /path/to/processed/dataset

* dataset_type: Type of dataset being used.

    * Choices: unlearncanvas, i2p
    * Type: str
    * Example: i2p

* template: Type of template for mask generation.

    * Choices: object, style, i2p
    * Type: str
    * Example: style

* template_name: Specific template name for the mask generation process.

    * Example Choices: self-harm, Abstractionism
    * Type: str
    * Example: Abstractionism

* threshold: Threshold value for mask generation to filter salient regions.

    * Type: float
    * Example: 0.5

**Output Configuration**

These parameters specify the directory where the results are saved.

* output_dir: Directory where the generated masks will be saved.

    * Type: str
    * Example: outputs/saliency_unlearning/masks


**Training Configuration**

These parameters control the training process for mask generation.

* lr: Learning rate used for training the masking algorithm.

    * Type: float
    * Example: 0.00001

* devices: CUDA devices used for training, specified as a comma-separated list.

    * Type: str
    * Example: 0

* use_sample: Flag indicating whether to use a sample dataset for training and mask generation.

    * Type: bool
    * Example: True


### Description of Arguments used to train saliency unlearning.

The following configs are used to fine-tune the Stable Diffusion model to perform saliency-based unlearning. This script relies on a configuration class `SaliencyUnlearningConfig`  and supports additional runtime arguments for further customization. Below is a detailed description of each argument:

**General Arguments**

* alpha: Guidance scale used to balance the loss components during training.
    
    * Type: float
    * Example: 0.1

* epochs: Number of epochs to train the model.
    
    * Type: int
    * Example: 5

* train_method: Specifies the training method or strategy to be used.

    * Choices: noxattn, selfattn, xattn, full, notime, xlayer, selflayer
    * Type: str
    * Example: noxattn

* model_config_path: Path to the model configuration YAML file for Stable Diffusion.
    
    * Type: str
    * Example: 'mu/algorithms/saliency_unlearning/configs/model_config.yaml'


**Dataset Arguments**

* raw_dataset_dir: Path to the directory containing the raw dataset, organized by themes and classes.

    * Type: str
    * Example: 'path/raw_dataset/'

* processed_dataset_dir: Path to the directory where the processed dataset will be saved.

    * Type: str
    * Example: 'path/processed_dataset_dir'

* dataset_type: Specifies the type of dataset to use for training.

    * Choices: unlearncanvas, i2p
    * Type: str
    * Example: i2p

* template: Specifies the template type for training.

    * Choices: object, style, i2p
    * Type: str
    * Example: style

* template_name: Name of the specific template used for training.

    * Example Choices: self-harm, Abstractionism
    * Type: str
    * Example: Abstractionism


**Output Arguments**

* output_dir: Directory where the fine-tuned model and training outputs will be saved.

    * Type: str
    * Example: 'output/folder_name'

* mask_path: Path to the saliency mask file used during training.

    * Type: str
    * Example: 



