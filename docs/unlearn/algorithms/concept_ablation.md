
# Concept Ablation Algorithm for Machine Unlearning

This repository provides an implementation of the Concept Ablation algorithm for machine unlearning in Stable Diffusion models. The Concept Ablation algorithm enables the removal of specific concepts or styles from a pre-trained model without the need for retraining from scratch.

---

## Installation

#### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

Before installing the unlearn_diff package, follow these steps to set up your environment correctly. These instructions ensure compatibility with the required dependencies, including Python, PyTorch, and ONNX Runtime.


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

### Example usage for quick canvas dataset

Add the following code snippet to a python script `trainer.py`. Run the script using `python trainer.py`.

```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    concept_ablation_train_mu,
    ConceptAblationConfig,
)

if __name__ == "__main__":

    concept_ablation_train_mu.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_mu,
        config_path="mu/algorithms/concept_ablation/configs/train_config.yaml",
        ckpt_path="machine_unlearning/models/compvis/style50/compvis.ckpt",
        prompts="mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt",
        output_dir="/opt/dlami/nvme/outputs",
        template_name = "Abstractionism", #concept to erase
        dataset_type = "unlearncanvas" ,
        use_sample = True, #train on sample dataset
        # devices="1",
    )
    algorithm.run()
```


### Example usage for i2p dataset

Add the following code snippet to a python script `trainer.py`. Run the script using `python trainer.py`.

```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    concept_ablation_train_i2p,
    ConceptAblationConfig,
)

if __name__ == "__main__":

    concept_ablation_train_i2p.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_i2p,
        config_path="mu/algorithms/concept_ablation/configs/train_config.yaml",
        raw_dataset_dir = "data/i2p-dataset/sample",
        ckpt_path="models/compvis/style50/compvis.ckpt",
        prompts="mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt",
        output_dir="/opt/dlami/nvme/outputs",
        template_name = "self-harm", #concept to erase
        dataset_type = "i2p" ,
        use_sample = True, #train on sample dataset
        # devices="1",
    )
    algorithm.run()
```


## Notes

1. Ensure all dependencies are installed as per the environment file.
2. The training process generates logs in the `logs/` directory for easy monitoring.
3. Use appropriate CUDA devices for optimal performance during training.
4. Regularly verify dataset and model configurations to avoid errors during execution.
---


## Directory Structure

- `algorithm.py`: Core implementation of the Concept Ablation Algorithm.
- `configs/`: Configuration files for training and generation.
- `data_handler.py`: Data handling and preprocessing.
- `scripts/train.py`: Script to train the Concept Ablation Algorithm.
- `callbacks/`: Custom callbacks for logging and monitoring training.
- `utils.py`: Utility functions.



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




