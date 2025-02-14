# Forget Me Not Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The Forget Me Not algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

### Installation
```
pip install unlearn_diff
```
### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

```bash
conda --version
```
### Create environment:
```
create_env <algorithm_name>
```
eg: ```create_env forget_me_not```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate forget_me_not```

The <algorithm_name> has to be one of the folders in the `mu/algorithms` folder.

### Downloading data and models.
After you install the package, you can use the following commands to download.

1. **Dataset**:
  - **i2p**:
    - **Sample**:
     ```
     download_data sample i2p
     ```
    - **Full**:
     ```
     download_data full i2p
     ```
  - **quick_canvas**:
    - **Sample**:
     ```
     download_data sample quick_canvas
     ```
    - **Full**:
     ```
     download_data full quick_canvas
     ```

2. **Model**:
  - **compvis**:
    ```
    download_model compvis
    ```
  - **diffuser**:
    ```
    download_model diffuser
    ```

**Verify the Downloaded Files**

After downloading, verify that the datasets have been correctly extracted:
```bash
ls -lh ./data/i2p-dataset/sample/
ls -lh ./data/quick-canvas-dataset/sample/
```
---

## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

1. **Train a Text Inversion**

```python

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_ti_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_ti_mu,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ), 
    steps=10
)
algorithm.run(train_type="train_ti")
```

**Running the Script in Offlikne Mode**

```bash
WANDB_MODE=offline python my_trainer_ti.py
```

2. **Perform Unlearning**

Before running the `train_attn` script, update the `ti_weights_path` parameter in the configuration file to point to the output generated from the Text Inversion (train_ti.py) stage

```python
from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_attn_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_attn_mu,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ),
    steps=10,
    ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors"
)
algorithm.run(train_type="train_attn")
```

**Running the Script in Offlikne Mode**

```bash
WANDB_MODE=offline python my_trainer_attn.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

## Directory Structure

- `algorithm.py`: Implementation of the Forget Me NotAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Forget Me NotModel class.
- `scripts/train.py`: Script to train the Forget Me Not algorithm.
- `trainer.py`: Implementation of the Forget Me NotTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---

**This method involves two stages:**

1. **Train a Text Inversion**: The first stage involves training a Text Inversion. Refer to the script [`train_ti.py`](mu/algorithms/forget_me_not/scripts/train_ti.py) for details and implementation. It uses `train_ti_config.yaml` as config file.

2. **Perform Unlearning**: The second stage uses the outputs from the first stage to perform unlearning. Refer to the script [`train_attn.py`](mu/algorithms/forget_me_not/scripts/train_attn.py) for details and implementation. It uses `train_attn_config.yaml` as config file.



### Description of Arguments in train_ti_config.yaml

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Training Configuration**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **steps**: Number of training steps.
- **lr**: Learning rate for the training optimizer.
- **weight_decay_ti**: Weight decay for Text Inversion training.
- **seed**: Random seed for reproducibility.
- **placeholder_tokens**: Tokens used as placeholders during training.
- **placeholder_token_at_data**: Placeholders used in the dataset for Text Inversion training.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_batch_size**: Batch size for training.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).



### Description of Arguments in train_attn_config.yaml

### Key Parameters

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Text Inversion**

- **use_ti**: Boolean indicating whether to use Text Inversion weights.
- **ti_weights_path**: File path to the Text Inversion model weights.

**Tokens**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **placeholder_tokens**: Tokens used as placeholders during training.

**Training Configuration**

- **mixed_precision**: Precision type to use during training (e.g., `fp16` or `fp32`).
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_text_encoder**: Boolean to enable or disable training of the text encoder.
- **enable_xformers_memory_efficient_attention**: Boolean to enable memory-efficient attention mechanisms.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **allow_tf32**: Boolean to allow TensorFloat-32 computation for faster training.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **train_batch_size**: Batch size for training.
- **use_8bit_adam**: Boolean to enable or disable 8-bit Adam optimizer.
- **adam_beta1**: Beta1 parameter for the Adam optimizer.
- **adam_beta2**: Beta2 parameter for the Adam optimizer.
- **adam_weight_decay**: Weight decay for the Adam optimizer.
- **adam_epsilon**: Epsilon value for the Adam optimizer.
- **size**: Image resolution size for training.
- **with_prior_preservation**: Boolean indicating whether to use prior preservation during training.
- **num_train_epochs**: Number of training epochs.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.
- **lr_num_cycles**: Number of cycles for learning rate scheduling.
- **lr_power**: Exponent to control the shape of the learning rate curve.
- **max-steps**: Maximum number of training steps.
- **no_real_image**: Boolean to skip using real images in training.
- **max_grad_norm**: Maximum norm for gradient clipping.
- **checkpointing_steps**: Number of steps between model checkpoints.
- **set_grads_to_none**: Boolean to set gradients to None instead of zeroing them out.
- **lr**: Learning rate for the training optimizer.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).

**Miscellaneous**

- **only-xa**: Boolean to enable additional configurations specific to the XA pipeline.



