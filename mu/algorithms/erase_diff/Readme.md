# ESD Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The erasediff algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### 1. Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate mu_erase_diff
```


## Usage

To train the erase_diff algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m algorithms.erase_diff.scripts.train \
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


 python -m algorithms.erase_diff.scripts.train --config_path "algorithms/erase_diff/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/data/results/erase_diff/models" --theme "Abstractionism" --class "Architectures" --use_sample 

