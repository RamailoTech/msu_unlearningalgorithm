# Saliency Unlearning Algorithm for Machine Unlearning

This repository provides an implementation of the Saliency Unlearning algorithm for machine unlearning in Stable Diffusion models. The Saliency Unlearning algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.


## Usage

Before training saliency unlearning algorithm you need to generate mask. Use the following code snippet to generate mask.

**Step 1: Generate mask**

**using unlearn canvas dataset**

```python
from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_generate_mask_mu

generate_mask = MaskingAlgorithm(
    saliency_unlearning_generate_mask_mu,
    ckpt_path = "models/compvis/style50/compvis.ckpt",
    raw_dataset_dir = "data/quick-canvas-dataset/sample",
    dataset_type = "unlearncanvas",
    use_sample = True, #to use sample dataset
    output_dir =  "outputs/saliency_unlearning/masks", #output path to save mask
    template_name = "Abstractionism",
    template = "style"
    )

if __name__ == "__main__":
    generate_mask.run()
```


**using i2p dataset**

```python
from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_generate_mask_i2p

generate_mask = MaskingAlgorithm(
    saliency_unlearning_generate_mask_i2p,
    ckpt_path = "models/compvis/style50/compvis.ckpt",
    raw_dataset_dir = "data/quick-canvas-dataset/sample",
    dataset_type = "unlearncanvas",
    use_sample = True, #to use sample dataset
    output_dir =  "outputs/saliency_unlearning/masks", #output path to save mask
    template_name = "self-harm",
    template = "i2p"
    )

if __name__ == "__main__":
    generate_mask.run()
```


### Run Train

**Using  quick canvas dataset**

To train the saliency unlearning algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

**Example Code**
```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearningAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_mu,
)

algorithm = SaliencyUnlearningAlgorithm(
    saliency_unlearning_train_mu,
    output_dir="/opt/dlami/nvme/outputs",
    dataset_type = "unlearncanvas"
    template_name = "Abstractionism", #concept to erase
    template = "style",
    use_sample = True #to run on sample dataset.
)
algorithm.run()
```

**Using i2p dataset**

```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearningAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_i2p,
)

algorithm = SaliencyUnlearningAlgorithm(
    saliency_unlearning_train_i2p,
    raw_dataset_dir = "data/i2p-dataset/sample",
    output_dir="/opt/dlami/nvme/outputs",
    template_name = "self-harm", #concept to erase
    template = "style",
    dataset_type = "i2p",
    use_sample = True #to run on sample dataset.
)
algorithm.run()
```

**Run on your own dataset**

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
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearnAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_mu,
)

algorithm = SaliencyUnlearnAlgorithm(
    saliency_unlearning_train_mu,
    raw_dataset_dir=(
        "data/generic" #replace with your own generated path
    ),
    ckpt_path="models/compvis/style50/compvis.ckpt",
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

---
<br>

**The unlearning has two stages:**

1. Generate the mask 

2. Unlearn the weights.

<br>

