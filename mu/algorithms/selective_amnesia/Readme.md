# Selective Amnesia Algorithm for Machine Unlearning

This repository provides an implementation of the Selective Amnesia algorithm for machine unlearning in Stable Diffusion models. The Selective Amnesia algorithm focuses on removing specific concepts or styles from a pre-trained model while retaining the rest of the knowledge.

---

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
eg: ```create_env selective_amnesia```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate selective_amnesia```

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

To train the Selective Amnesia algorithm to remove specific concepts or styles from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

1. First download the full_fisher_dict.pkl file.
```
wget https://huggingface.co/ajrheng/selective-amnesia/resolve/main/full_fisher_dict.pkl
```

2. Run the script 
```
python -m mu.algorithms.selective_amnesia.scripts.train --config_path mu/algorithms/selective_amnesia/configs/train_config.yaml --full_fisher_dict_pkl_path /path/full_fisher_dict.pkl
```

### Running the Training Script in Offline Mode

```
WANDB_MODE=offline python -m mu.algorithms.selective_amnesia.scripts.train --config_path mu/algorithms/selective_amnesia/configs/train_config.yaml --full_fisher_dict_pkl_path /path/full_fisher_dict.pkl
```

### Overriding Configuration via Command Line

You can override configuration parameters by passing them directly as arguments during runtime.

**Example Usage with Command-Line Arguments:**

```bash
python -m mu.algorithms.selective_amnesia.scripts.train \
--config_path mu/algorithms/selective_amnesia/configs/train_config.yaml \
--train_batch_size 8 \
--max_epochs 100 \
--devices 0,1 \
--output_dir outputs/experiment_2
```

**Explanation:**
* `--config_path`: Specifies the YAML configuration file.
* `--train_batch_size`: Overrides the training batch size to 8.
* `--max_epochs`: Updates the maximum number of training epochs to 100.
* `--devices`: Specifies the GPUs (e.g., device 0 and 1).
* `--output_dir`: Sets a custom output directory for the experiment.

---

## Directory Structure

- `algorithm.py`: Core implementation of the Selective Amnesia Algorithm.
- `configs/`: Configuration files for training and generation.
- `data_handler.py`: Data handling and preprocessing.
- `scripts/train.py`: Script to train the Selective Amnesia Algorithm.
- `callbacks/`: Custom callbacks for logging and monitoring training.
- `utils.py`: Utility functions.

---

## How It Works

1. **Default Configuration:** Loads values from the specified YAML file (`--config_path`).
2. **Command-Line Overrides:** Updates the configuration with values provided as command-line arguments.
3. **Training Execution:** Initializes the `SelectiveAmnesiaAlgorithm` and trains the model using the provided dataset, model checkpoint, and configuration.
4. **Output:** Saves the fine-tuned model and logs training metrics in the specified output directory.

---

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



