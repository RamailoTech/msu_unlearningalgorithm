# ScissorHands Algorithm for Machine Unlearning

This repository provides an implementation of the scissor hands algorithm for machine unlearning in Stable Diffusion models. The scissor hands algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.


## Usage

To train the ScissorHands algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**

**Using quick canvas dataset**

```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import (
    scissorhands_train_mu,
)

algorithm = ScissorHandsAlgorithm(
    scissorhands_train_mu,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
    dataset_type = "unlearncanvas",
    template = "style",
    template_name = "Abstractionism",
    use_sample = True # to train on sample dataset
)
algorithm.run()
```

**Using i2p dataset**

```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import (
    scissorhands_train_i2p,
)

algorithm = ScissorHandsAlgorithm(
    scissorhands_train_i2p,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir = "data/i2p-dataset/sample",
    output_dir="/opt/dlami/nvme/outputs",
    use_sample = True, # to train on sample dataset
    dataset_type = "i2p",
    template_name = "self-harm"
)
algorithm.run()
```

**Use your own dataset**
   
**Step-1: Generate your own dataset**

```bash
generate_images_for_prompts --model_path models/diffuser/style50 --csv_path data/prompts/generic_data.csv
```

Note:

* generate_images_for_prompts: This command invokes the image generation script. It uses a diffusion model to generate images based on textual prompts.

* --model_path: Specifies the path to the diffusion model to be used for image generation. In this example, the model is located at models/diffuser/style50.

* --csv_path: Provides the path to a CSV file containing the prompts. Each prompt in this CSV will be used to generate an image, allowing you to build a dataset tailored to your needs.

**Step-2: Train on your own dataset**

```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import scissorhands_train_mu

algorithm = ScissorHandsAlgorithm(
    scissorhands_train_mu,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir="data/generic_data", #use your own generated path
    output_dir="/opt/dlami/nvme/outputs",
    dataset_type = "generic",
    template_name = "self-harm"
)
algorithm.run()
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python my_trainer.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

### Directory Structure

- `algorithm.py`: Implementation of the ScissorHandsAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the ScissorHandsModel class.
- `scripts/train.py`: Script to train the ScissorHands algorithm.
- `trainer.py`: Implementation of the ScissorHandsTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---
