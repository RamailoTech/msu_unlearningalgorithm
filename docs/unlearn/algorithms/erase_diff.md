# EraseDiff Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The erasediff algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.


## Run Train using quick canvas dataset
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import (
    erase_diff_train_mu,
)

algorithm = EraseDiffAlgorithm(
    erase_diff_train_mu,
    ckpt_path="UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    template_name = "Abstractionism", #concept to erase
    dataset_type = "unlearncanvas" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/erase_diff/finetuned_models" #output dir to save finetuned models
)
algorithm.run()
```



## Run Train using i2p dataset
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import (
    erase_diff_train_i2p,
)

algorithm = EraseDiffAlgorithm(
    erase_diff_train_i2p,
    ckpt_path="UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir = "data/i2p-dataset/sample",
    template_name = "self-harm", #concept to erase
    dataset_type = "i2p" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/erase_diff/finetuned_models" #output dir to save finetuned models
)
algorithm.run()
```

**Use your own dataset for unlearning**

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
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

algorithm = EraseDiffAlgorithm(
    erase_diff_train_mu,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir="data/generic", #replace with your own generated path
    train_method="noxattn",
    dataset_type="generic",
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

- `algorithm.py`: Implementation of the EraseDiffAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the EraseDiffModel class.
- `scripts/train.py`: Script to train the EraseDiff algorithm.
- `trainer.py`: Implementation of the EraseDiffTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---

