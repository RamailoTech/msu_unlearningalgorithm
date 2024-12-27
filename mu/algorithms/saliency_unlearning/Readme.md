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



### Description of Arguments mask_config.yaml

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

    * Choices: self-harm, Abstractionism
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
    * Example: 


## Usage

To train the mu_saliency_unlearning algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m algorithms.saliency_unlearning.scripts.train \
    --train_method xattn \
    --theme "Your_Theme" \
    --ckpt_path "path/to/your/model.ckpt" \
    --config_path "path/to/your_config.yaml" \
    --output_dir "path/to/your_output_dir"
```

**Replace the placeholders with your own values:**

- `Your_Theme`: The concept or style you want the model to unlearn (e.g., `"Van_Gogh_Style"`).
- `path/to/your/model.ckpt`: Path to your pre-trained Stable Diffusion model checkpoint.
- `configs/train_esd.yaml`: Path to the ESD training configuration file.


## Directory Structure

- `algorithm.py`: Implementation of the ESDAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the ESDModel class.
- `sampler.py`: Implementation of the ESDSampler class.
- `scripts/train.py`: Script to train the ESD algorithm.
- `trainer.py`: Implementation of the ESDTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class


For running the masking script 

 python -m algorithms.saliency_unlearning.scripts.generate_mask --config_path "algorithms/saliency_unlearning/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --theme "Abstractionism" --class "Architectures" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/saliency_unlearning/data/masks" 


 python -m algorithms.saliency_unlearning.scripts.train --config_path "algorithms/saliency_unlearning/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/data/results/saliency_unlearning/models" --theme "Abstractionism" --classes "Architectures" --mask_path "/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/saliency_unlearning/data/masks/Abstractionism/0.5.pt" --use_sample 

Running generate_mask

 python scripts/generate_mask.py \
    --ckpt_path "path/to/checkpoint.ckpt" \
    --config_path "configs/saliency_unlearn_config.yaml" \
    --theme "Abstractionism" \
    --forget_data_dir "data" \
    --remain_data_dir "data/Seed_Images" \
    --output_dir "output/masks" \
    --threshold 0.4 \
    --c_guidance 7.5 \
    --batch_size 4 \
    --image_size 512 \
    --num_timesteps 1000 \
    --lr 1e-5



python -m mu.algorithms.saliency_unlearning.scripts.generate_mask \
--config_path mu/algorithms/saliency_unlearning/configs/mask_config.yaml

python -m mu.algorithms.saliency_unlearning.scripts.train \
--config_path mu/algorithms/saliency_unlearning/configs/train_config.yaml

