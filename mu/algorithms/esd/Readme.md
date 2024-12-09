# ESD Algorithm for Machine Unlearning

This repository provides an implementation of the ESD algorithm for machine unlearning in Stable Diffusion models. The ESD algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### 1. Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate mu_esd
```


## Usage

To train the ESD algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m algorithms.esd.scripts.train \
    --train_method xattn \
    --theme "Your_Theme" \
    --ckpt_path "path/to/your/model.ckpt" \
    --config_path "path/to/your_config" \
    ----output_dir "path/to/your_output_dir"
```

**Replace the placeholders with your own values:**

- `Your_Theme`: The concept or style you want the model to unlearn (e.g., `"Van_Gogh_Style"`).
- `path/to/your/model.ckpt`: Path to your pre-trained Stable Diffusion model checkpoint.
- `configs/train_esd.yaml`: Path to the ESD training configuration file.

### Optional Arguments

- `--iterations`: Number of training iterations (default: `1000`).
- `--lr`: Learning rate for training (default: `1e-5`).
- `--start_guidance`: Guidance scale for the initial image generation (default: `3`).
- `--negative_guidance`: Guidance scale for the negative prompt (default: `1`).
- `--devices`: CUDA devices to use (default: `'0,0'`).
- `--output_dir`: Directory to save the trained model (default: `'results'`).


## Directory Structure

- `algorithm.py`: Implementation of the ESDAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `constants/const.py`: Constants used throughout the project.
- `model.py`: Implementation of the ESDModel class.
- `sampler.py`: Implementation of the ESDSampler class.
- `scripts/train.py`: Script to train the ESD algorithm.
- `trainer.py`: Implementation of the ESDTrainer class.
- `utils.py`: Utility functions used in the project.


python -m algorithms.esd.scripts.train --train_method xattn --theme "Abstractionism" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --config_path "algorithms/esd/configs/train_esd.yaml" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/data/results/esd/models"