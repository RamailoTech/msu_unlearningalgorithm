
# Concept Ablation Algorithm for Machine Unlearning

This repository provides an implementation of the Concept Ablation algorithm for machine unlearning in Stable Diffusion models. The Concept Ablation algorithm enables the removal of specific concepts or styles from a pre-trained model without the need for retraining from scratch.

---

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/concept_ablation/environment.yaml
```

```bash
conda activate mu_concept_ablation
```

### Download Models

To download the [models](https://huggingface.co/nebulaanish/unlearn_models/tree/main), use the following commands:

1. **Compvis Model (Size: 3.84 GB):**
    * Make it executable:
        ```bash
        chmod +x scripts/download_models.sh
        ```
    * Run the script:
        ```bash
        scripts/download_models.sh compvis
        ```

2. **Diffuser Model (Size: 24.1 GB):**
    * Make it executable:
        ```bash
        chmod +x scripts/download_models.sh
        ```
    * Run the script:
        ```bash
        scripts/download_models.sh diffuser
        ```

**Verify Downloads**

After downloading, verify the extracted files in their respective directories:
```bash
ls -lh ../models/compvis/
ls -lh ../models/diffuser/
```

### Download Datasets

1. **Download Unlearn Canvas Dataset:**
    * Make it executable:
        ```bash
        chmod +x scripts/download_quick_canvas_dataset.sh
        ```
    * Download the sample dataset (smaller size):
        ```bash
        scripts/download_quick_canvas_dataset.sh sample
        ```
    * Download the full dataset:
        ```bash
        scripts/download_quick_canvas_dataset.sh full
        ```

2. **Download the i2p Dataset:**
    * Make it executable:
        ```bash
        chmod +x scripts/download_i2p_dataset.sh
        ```
    * Download the sample dataset (smaller size):
        ```bash
        scripts/download_i2p_dataset.sh sample
        ```
    * Download the full dataset:
        ```bash
        scripts/download_i2p_dataset.sh full
        ```

**Verify the Downloaded Files**

After downloading, verify that the datasets have been correctly extracted:
```bash
ls -lh ./data/i2p-dataset/sample/
ls -lh ./data/quick-canvas-dataset/sample/
```

---

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

---

## Usage

To train the Concept Ablation algorithm to remove specific concepts or styles from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

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

