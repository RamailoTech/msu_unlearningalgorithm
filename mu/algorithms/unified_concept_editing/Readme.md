# Unified Concept Editing Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The erasediff algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### 1. Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate mu_uce
```


## Usage

To train the Unified Concept Editing algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m algorithms.unified_concept_editing.scripts.train \
    --theme "Your_Theme" \
    --ckpt_path "path/to/your/model.ckpt" \
    --config_path "path/to/your_config.yaml" \
    --output_dir "path/to/your_output_dir"
```

**Replace the placeholders with your own values:**

- `Your_Theme`: The concept or style you want the model to unlearn (e.g., `"Van_Gogh_Style"`).
- `path/to/your/model`: Path to your pre-trained Stable Diffusion model checkpoint.
- `configs/train_config.yaml`: Path to the Unified Concept Editing training configuration file.


## Directory Structure

- `algorithm.py`: Implementation of the Unified Concept EditingAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Unified Concept EditingModel class.
- `scripts/train.py`: Script to train the Unified Concept Editing algorithm.
- `trainer.py`: Implementation of the Unified Concept EditingTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class


 python -m algorithms.unified_concept_editing.scripts.train --config_path "algorithms/unified_concept_editing/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/data/results/erase_diff/models" --theme "Abstractionism" --class "Architectures" --use_sample 

