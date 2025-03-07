# Semi Permeable Membrane Algorithm for Machine Unlearning

This repository provides an implementation of the semipermeable membrane algorithm for machine unlearning in Stable Diffusion models. The semipermeable membrane algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Usage

To train the Semi Permeable Membrane algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**

**Using quick canvas dataset**


```python

from mu.algorithms.semipermeable_membrane.algorithm import (
    SemipermeableMembraneAlgorithm,
)
from mu.algorithms.semipermeable_membrane.configs import (
    semipermiable_membrane_train_mu,
    SemipermeableMembraneConfig,
)

algorithm = SemipermeableMembraneAlgorithm(
    semipermiable_membrane_train_mu,
    output_dir="/opt/dlami/nvme/outputs",
    train={"iterations": 2},
    use_sample = True # to run on sample dataset
    
)
algorithm.run()
```

**Using quick canvas dataset**


```python

from mu.algorithms.semipermeable_membrane.algorithm import (
    SemipermeableMembraneAlgorithm,
)
from mu.algorithms.semipermeable_membrane.configs import (
    semipermiable_membrane_train_i2p,
    SemipermeableMembraneConfig,
)

algorithm = SemipermeableMembraneAlgorithm(
    semipermiable_membrane_train_i2p,
    output_dir="/opt/dlami/nvme/outputs",
    train={"iterations": 2},
    use_sample = True # to run on sample dataset
    dataset_type = "i2p",
    template_name = "self-harm",
    
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
## Directory Structure

- `algorithm.py`: Implementation of the Semi Permeable MembraneAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Semi Permeable MembraneModel class.
- `scripts/train.py`: Script to train the Semi Permeable Membrane algorithm.
- `trainer.py`: Implementation of the Semi Permeable MembraneTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---
