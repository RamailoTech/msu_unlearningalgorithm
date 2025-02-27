# Selective Amnesia Algorithm for Machine Unlearning

This repository provides an implementation of the Selective Amnesia algorithm for machine unlearning in Stable Diffusion models. The Selective Amnesia algorithm focuses on removing specific concepts or styles from a pre-trained model while retaining the rest of the knowledge.

---

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

To train the Selective Amnesia algorithm to remove specific concepts or styles from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.


**First download the full_fisher_dict.pkl file.**
```
wget https://huggingface.co/ajrheng/selective-amnesia/resolve/main/full_fisher_dict.pkl
```


### Run train

Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Using quick canvas dataset**


```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    selective_amnesia_config_quick_canvas,
)

algorithm = SelectiveAmnesiaAlgorithm(
    selective_amnesia_config_quick_canvas,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    dataset_type = "unlearncanvas",
    template = "style",
    template_name = "Abstractionism",
    use_sample = True # to run on sample dataset

)
algorithm.run()

```


**Using i2p dataset**


```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    selective_amnesia_config_i2p,
)

algorithm = SelectiveAmnesiaAlgorithm(
    selective_amnesia_config_i2p,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/i2p/sample"
    ),
    dataset_type = "i2p",
    template_name = "self-harm",
    use_sample = True # to run on sample dataset
)
algorithm.run()

```


```bash
WANDB_MODE=offline python my_trainer.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

## Notes

1. Ensure all dependencies are installed as per the environment file.
2. The training process generates logs in the `logs/` directory for easy monitoring.
3. Use appropriate CUDA devices for optimal performance during training.
4. Regularly verify dataset and model configurations to avoid errors during execution.
---

## Configuration File (`train_config.yaml`)

### Training Parameters

* **seed:** Random seed for reproducibility.
    * Type: int
    * Example: 23

* **scale_lr:** Whether to scale the base learning rate.
    * Type: bool
    * Example: True

### Model Configuration

* **model_config_path:** Path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/model_config.yaml"

* **ckpt_path:** Path to the Stable Diffusion model checkpoint.
    * Type: str
    * Example: "/path/to/compvis.ckpt"

* **full_fisher_dict_pkl_path:** Path to the full fisher dict pkl file
    * Type: str
    * Example: "full_fisher_dict.pkl"

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
    * Example: "outputs/selective_amnesia/finetuned_models"

### Device Configuration

* **devices:** CUDA devices for training (comma-separated).
    * Type: str
    * Example: "0"

### Data Parameters

* **train_batch_size:** Batch size for training.
    * Type: int
    * Example: 4

* **val_batch_size:** Batch size for validation.
    * Type: int
    * Example: 6

* **num_workers:** Number of worker threads for data loading.
    * Type: int
    * Example: 4

* **forget_prompt:** Prompt to specify the style or concept to forget.
    * Type: str
    * Example: "An image in Artist_Sketch style"

### Lightning Configuration

* **max_epochs:** Maximum number of epochs for training.
    * Type: int
    * Example: 50

* **callbacks:**
    * **batch_frequency:** Frequency for logging image batches.
        * Type: int
        * Example: 1

    * **max_images:** Maximum number of images to log.
        * Type: int
        * Example: 999

---



