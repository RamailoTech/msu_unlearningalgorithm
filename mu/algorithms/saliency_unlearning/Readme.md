# Saliency Unlearning Algorithm for Machine Unlearning

This repository provides an implementation of the Saliency Unlearning algorithm for machine unlearning in Stable Diffusion models. The Saliency Unlearning algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/saliency_unlearning/environment.yaml -n mu_saliency_unlearning
```

```bash
conda activate mu_saliency_unlearning
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


<br>

**The unlearning has two stages:**

1. Generate the mask 

2. Unlearn the weights.

<br>

### Description of Arguments in mask_config.yaml

The `config/mask_config.yaml` file is a configuration file for generating saliency masks using the `scripts/generate_mask.py` script. It defines various parameters related to the model, dataset, output, and training. Below is a detailed description of each section and parameter:

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


### Description of Arguments train_config.yaml

The `scripts/train.py` script is used to fine-tune the Stable Diffusion model to perform saliency-based unlearning. This script relies on a configuration file (`config/train_config.yaml`) and supports additional runtime arguments for further customization. Below is a detailed description of each argument:

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


## Usage

To train the saliency unlearning algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

**Step 1: Generate mask**

```bash
python -m mu.algorithms.saliency_unlearning.scripts.generate_mask \
--config_path mu/algorithms/saliency_unlearning/configs/mask_config.yaml
```

**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.saliency_unlearning.scripts.generate_mask \
--config_path mu/algorithms/saliency_unlearning/configs/mask_config.yaml
```

**Step 2: Unlearn the weights**

- Add the generated mask path to the `train_config.yaml` file or you can override it by passing them directly as arguments during runtime.

- Run the script:

```bash
python -m mu.algorithms.saliency_unlearning.scripts.train \
--config_path mu/algorithms/saliency_unlearning/configs/train_config.yaml
```

**Running the Script in Offline Mode**
```bash
WANDB_MODE=offline python -m mu.algorithms.saliency_unlearning.scripts.train \
--config_path mu/algorithms/saliency_unlearning/configs/train_config.yaml
```


**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.


**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.saliency_unlearning.scripts.train \
--config_path mu/algorithms/saliency_unlearning/configs/train_config.yaml \
--mask_path /path/to/mask.pt \
--alpha 0.1 \
--epochs 10 \
--raw_dataset_dir /path/to/raw_dataset \
--output_dir outputs/experiment_1
```

**Explanation of the Example**

* --config_path: Specifies the YAML configuration file to load default values.
* --mask_path: Path of the generated mask.
* --alpha: Sets the guidance strength for the starting image to 0.2.
* --epochs: Increases the number of training epochs to 10.
* --lr: Updates the learning rate to 1e-4.
* --raw_dataset_dir: Changes the raw dataset directory.
* --output_dir: Sets a custom output directory for this run.

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


