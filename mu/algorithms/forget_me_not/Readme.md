# Forget Me Not Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The Forget Me Not algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### 1. Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate mu_forget_me_not
```


## Usage

To train the forget_me_not algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m algorithms.forget_me_not.scripts.train \
    --train_method xattn \
    --theme "Your_Theme" \
    --ckpt_path "path/to/your/model.ckpt" \
    --config_path "path/to/your_config.yaml" \
    --output_dir "path/to/your_output_dir"
```

**Replace the placeholders with your own values:**

- `Your_Theme`: The concept or style you want the model to unlearn (e.g., `"Van_Gogh_Style"`).
- `path/to/your/model.ckpt`: Path to your pre-trained Stable Diffusion model checkpoint.
- `configs/train_Forget Me Not.yaml`: Path to the Forget Me Not training configuration file.


## Directory Structure

- `algorithm.py`: Implementation of the Forget Me NotAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Forget Me NotModel class.
- `sampler.py`: Implementation of the Forget Me NotSampler class.
- `scripts/train.py`: Script to train the Forget Me Not algorithm.
- `trainer.py`: Implementation of the Forget Me NotTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

 python -m algorithms.forget_me_not.scripts.train_ti --config_path "algorithms/forget_me_not/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --theme "Abstractionism" --class "Architectures" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/forget_me_not/data/masks" 


 python -m algorithms.forget_me_not.scripts.train_attn --config_path "algorithms/forget_me_not/config/train_config.yaml" --ckpt_path "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" --output_dir "/home/ubuntu/Projects/msu_unlearningalgorithm/data/results/forget_me_not/models" --theme "Abstractionism" --classes "Architectures" --mask_path "/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/forget_me_not/data/masks/Abstractionism/0.5.pt" --use_sample 
