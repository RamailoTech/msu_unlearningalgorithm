# Unified Concept Editing Algorithm for Machine Unlearning

This repository provides an implementation of the unified concept editing algorithm for machine unlearning in Stable Diffusion models. The unified concept editing algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/unified_concept_editing/environment.yaml -n mu_uce
```

```bash
conda activate conda env create -f mu/algorithms/unified_concept_editing/environment.yaml -n mu_uce

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

* **train_method**: Specifies the method of training for concept erasure.
    * Choices: ["full", "partial"]
    * Example: "full"

* **alpha**: Guidance strength for the starting image during training.
    * Type: float
    * Example: 0.1

* **epochs**: Number of epochs to train the model.
    * Type: int
    * Example: 10

* **lr**: Learning rate used for the optimizer during training.
    * Type: float
    * Example: 5e-5


**Model Configuration**
* **ckpt_path**: File path to the checkpoint of the Stable Diffusion model.
    * Type: str
    * Example: "/path/to/model_checkpoint.ckpt"

* **config_path**: File path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/config.yaml"

**Dataset Directories**

* **dataset_type**: Specifies the dataset type for the training process.
    * Choices: ["unlearncanvas", "i2p"]
    * Example: "unlearncanvas"

* **template**: Type of template to use during training.
    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* **template_name**: Name of the specific concept or style to be erased.
    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"

**Output Configurations**

* **output_dir**: Directory where the fine-tuned models and results will be saved.
    * Type: str
    * Example: "outputs/erase_diff/finetuned_models"

**Sampling and Image Configurations**

* **use_sample**: Flag to indicate whether a sample dataset should be used for training.
    * Type: bool
    * Example: True

* **guided_concepts**: Concepts to guide the editing process.
    * Type: str
    * Example: "Nature, Abstract"

* **technique**: Specifies the editing technique.
    * Choices: ["replace", "tensor"]
    * Example: "replace"

* **preserve_scale**: Scale for preservation during the editing process.
    * Type: float
    * Example: 0.5

* **preserve_number**: Number of items to preserve during editing.
    * Type: int
    * Example: 10

* **erase_scale**: Scale for erasure during the editing process.
    * Type: float
    * Example: 0.8

* **lamb**: Lambda parameter for controlling balance during editing.
    * Type: float
    * Example: 0.01

* **add_prompts**: Flag to indicate whether additional prompts should be used.
    * Type: bool
    * Example: True

**Device Configuration**

* **devices**: Specifies the CUDA devices to be used for training (comma-separated).
    * Type: str (Comma-separated)
    * Example: "0,1"




## Usage

To train the Unified Concept Editing algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

**Example Command**

```bash
python -m mu.algorithms.unified_concept_editing.scripts.train \
--config_path mu/algorithms/unified_concept_editing/configs/train_config.yaml
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.unified_concept_editing.scripts.train \
--config_path mu/algorithms/unified_concept_editing/configs/train_config.yaml
```

**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.


**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.unified_concept_editing.scripts.train \
--config_path mu/algorithms/unified_concept_editing/configs/train_config.yaml \
--train_method xattn \
--alpha 0.2 \
--devices 0,1 \
--output_dir outputs/experiment_1
```

**Explanation of the Example**

* --config_path: Specifies the YAML configuration file to load default values.
* --train_method: Overrides the training method ("xattn").
* --alpha: Sets the guidance strength for the starting image to 0.2.
* --devices: Specifies the GPUs (e.g., device 0 and 1) for training.
* --output_dir: Sets a custom output directory for this run.


**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.


### Directory Structure

- `algorithm.py`: Implementation of the ScissorHandsAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the ScissorHandsModel class.
- `scripts/train.py`: Script to train the ScissorHands algorithm.
- `trainer.py`: Implementation of the ScissorHandsTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class
