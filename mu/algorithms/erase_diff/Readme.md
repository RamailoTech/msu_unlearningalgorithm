# EraseDiff Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The erasediff algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/erase_diff/environment.yaml -n mu_erase_diff
```

```bash
conda activate mu_erase_diff
```

### Download models

To download [models](https://huggingface.co/nebulaanish/unlearn_models/tree/main), use the following commands <br>

1. Compvis (Size 3.84 GB):

    * Make it executable:

        `chmod +x scripts/download_models.sh`

    * Run the script:
        ```scripts/download_models.sh compvis```

2. Diffuser (24.1 GB): 

    * Make it executable:

        `chmod +x scripts/download_models.sh`

    * Run the script: 
        ```scripts/download_models.sh diffuser```

**Notes:**

1. The script ensures that directories are automatically created if they don’t exist.
2. The downloaded ZIP file will be extracted to the respective folder, and the ZIP file will be removed after extraction.


**Verify Downloads**

After downloading, you can verify the extracted files in their respective directories:

`ls -lh ../models/compvis/`

`ls -lh ../models/diffuser/`

### Download datasets

1. Download unlearn canvas dataset:

    * Make it executable:

        `chmod +x scripts/download_quick_canvas_dataset.sh`

    * Download the sample dataset (smaller size):

        `scripts/download_quick_canvas_dataset.sh sample`

    * Download the full dataset:

        `scripts/download_quick_canvas_dataset.sh full`

2. Download the i2p dataset

    * Make it executable:

        `chmod +x scripts/download_i2p_dataset.sh`

    * Download the sample dataset (smaller size):

        `scripts/download_i2p_dataset.sh sample`

    * Download the full dataset:

        `scripts/download_i2p_dataset.sh full`

**Notes:**

1. The script automatically creates the required directories if they don't exist.
2. Ensure curl and unzip are installed on your system.

**Verify the Downloaded files**

After downloading, verify that the datasets have been correctly extracted:

`ls -lh ./data/i2p-dataset/sample/`

`ls -lh ./data/quick-canvas-dataset/sample/`


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

* K_steps: Number of K optimization steps during training.

    * Type: int
    * Example: 2

* lr: Learning rate used for the optimizer during training.

    * Type: float
    * Example: 5e-5

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

* separator: String separator used to train multiple words separately, if applicable.

    * Type: str or null
    * Example: null

**Sampling and Image Configurations**

* image_size: Size of the training images (height and width in pixels).

    * Type: int
    * Example: 512

* interpolation: Interpolation method used for image resizing.

    * Choices: ["bilinear", "bicubic", "lanczos"]
    * Example: "bicubic"

* ddim_steps: Number of DDIM inference steps during training.

    * Type: int
    * Example: 50

* ddim_eta: DDIM eta parameter for stochasticity during sampling.

    * Type: float
    * Example: 0.0

**Device Configuration**

* devices: Specifies the CUDA devices to be used for training (comma-separated).

    * Type: str
    * Example: "0"


**Additional Flags**

* use_sample: Flag to indicate whether a sample dataset should be used for training.

    * Type: bool
    * Example: True

* num_workers: Number of worker threads for data loading.

    * Type: int
    * Example: 4

* pin_memory: Flag to enable pinning memory during data loading for faster GPU transfers.

    * Type: bool
    * Example: true




## Usage

To train the erase_diff algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

**Example Command**

```bash
python -m mu.algorithms.erase_diff.scripts.train \
--config_path mu/algorithms/erase_diff/configs/train_config.yaml
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.erase_diff.scripts.train \
--config_path mu/algorithms/erase_diff/configs/train_config.yaml
```

**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.


**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.erase_diff.scripts.train \
--config_path mu/algorithms/erase_diff/configs/train_config.yaml \
--train_method xattn \
--alpha 0.2 \
--epochs 10 \
--lr 1e-4 \
--devices 0,1 \
--raw_dataset_dir /path/to/raw_dataset \
--output_dir outputs/experiment_1
```

**Explanation of the Example**

* --config_path: Specifies the YAML configuration file to load default values.
* --train_method: Overrides the training method ("xattn").
* --alpha: Sets the guidance strength for the starting image to 0.2.
* --epochs: Increases the number of training epochs to 10.
* --lr: Updates the learning rate to 1e-4.
* --devices: Specifies the GPUs (e.g., device 0 and 1) for training.
* --raw_dataset_dir: Changes the raw dataset directory.
* --output_dir: Sets a custom output directory for this run.


**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.


### Directory Structure

- `algorithm.py`: Implementation of the EraseDiffAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the EraseDiffModel class.
- `scripts/train.py`: Script to train the EraseDiff algorithm.
- `trainer.py`: Implementation of the EraseDiffTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class




