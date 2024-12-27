# ESD Algorithm for Machine Unlearning

This repository provides an implementation of the ESD algorithm for machine unlearning in Stable Diffusion models. The ESD algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/esd/environment.yaml -n mu_esd
```

```bash
conda activate mu_esd
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

* ckpt_path: Path to the pre-trained Stable Diffusion model checkpoint.

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

## Usage

To train the ESD algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

**Example Command**

```bash
python -m mu.algorithms.esd.scripts.train \
--config_path mu/algorithms/esd/configs/train_config.yaml
```

**Running the Script in Offline Mode**
```bash
WANDB_MODE=offline python -m mu.algorithms.esd.scripts.train \
--config_path mu/algorithms/esd/configs/train_config.yaml
```


**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.


```bash
python mu/algorithms/esd/scripts/train.py \
    --config_path train_config.yaml \
    --train_method "xattn" \
    --start_guidance 0.1 \
    --negative_guidance 0.0 \
    --iterations 1000 \
    --lr 5e-5 
```


**Explanation of the Example**

* train_method: Specifies which model layers to update during training.
* start_guidance: Guidance scale for generating initial images.
* negative_guidance: Guidance scale for erasing the target concept.
* iterations: Number of training iterations (epochs).
* lr: Learning rate for the optimizer.

**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.



## Directory Structure

- `algorithm.py`: Implementation of the ESDAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `constants/const.py`: Constants used throughout the project.
- `model.py`: Implementation of the ESDModel class.
- `scripts/train.py`: Script to train the ESD algorithm.
- `trainer.py`: Implementation of the ESDTrainer class.
- `utils.py`: Utility functions used in the project.

