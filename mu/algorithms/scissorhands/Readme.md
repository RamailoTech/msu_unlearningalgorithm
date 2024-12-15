# Scissorhands Algorithm for Machine Unlearning

This repository provides an implementation of the scissorhands  algorithm for machine unlearning in Stable Diffusion models. The scissorhands algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### 1. Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate mu_scissorhands
```


## Usage

To train the scissorhands algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m algorithms.scissorhands.scripts.train \
    --train_method xattn \
    --theme "Your_Theme" \
    --ckpt_path "path/to/your/model.ckpt" \
    --config_path "path/to/your_config.yaml" \
    --output_dir "path/to/your_output_dir"
```

**Replace the placeholders with your own values:**

- `Your_Theme`: The concept or style you want the model to unlearn (e.g., `"Van_Gogh_Style"`).
- `path/to/your/model.ckpt`: Path to your pre-trained Stable Diffusion model checkpoint.
- `configs/train_Scissorhands.yaml`: Path to the Scissorhands training configuration file.


## Directory Structure

- `algorithm.py`: Implementation of the ScissorhandsAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the ScissorhandsModel class.
- `sampler.py`: Implementation of the ScissorhandsSampler class.
- `scripts/train.py`: Script to train the Scissorhands algorithm.
- `trainer.py`: Implementation of the ScissorhandsTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class


 python -m algorithms.erase_diff.scripts.train         --config_path "algorithms/scissorhands/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --theme Abstractionism --classes Architectures --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/data/results/scissorhands/models" --use_sample"

