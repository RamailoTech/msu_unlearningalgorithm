


**Step 1: Generate mask using unlearn canvas dataset**

Before training saliency unlearning algorithm you need to generate mask. Use the following code snippet to generate mask.


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


**Step 2: Unlearn Using  quick canvas dataset**

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



**The unlearning has two stages:**

1. Generate the mask 

2. Unlearn the weights.

<br>

