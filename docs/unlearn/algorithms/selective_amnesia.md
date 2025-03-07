# Selective Amnesia Algorithm for Machine Unlearning

This repository provides an implementation of the Selective Amnesia algorithm for machine unlearning in Stable Diffusion models. The Selective Amnesia algorithm focuses on removing specific concepts or styles from a pre-trained model while retaining the rest of the knowledge.


## Usage

To train the Selective Amnesia algorithm to remove specific concepts or styles from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.


**First download the full_fisher_dict.pkl file.**
```
wget https://huggingface.co/ajrheng/selective-amnesia/resolve/main/full_fisher_dict.pkl
```


### Run train

Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Using quick canvas dataset**


```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    selective_amnesia_config_quick_canvas,
)

algorithm = SelectiveAmnesiaAlgorithm(
    selective_amnesia_config_quick_canvas,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    dataset_type = "unlearncanvas",
    template = "style",
    template_name = "Abstractionism",
    use_sample = True # to run on sample dataset

)
algorithm.run()

```


**Using i2p dataset**


```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    selective_amnesia_config_i2p,
)

algorithm = SelectiveAmnesiaAlgorithm(
    selective_amnesia_config_i2p,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/i2p/sample"
    ),
    dataset_type = "i2p",
    template_name = "self-harm",
    use_sample = True # to run on sample dataset
)
algorithm.run()

```

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
    from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
    from mu.algorithms.selective_amnesia.configs import (
        selective_amnesia_config_quick_canvas,
    )

    algorithm = SelectiveAmnesiaAlgorithm(
        selective_amnesia_config_quick_canvas,
        ckpt_path="models/compvis/style50/compvis.ckpt", 
        raw_dataset_dir="data/generic", #use your own path
        dataset_type = "generic",
        template_name = "self-harm",
        replay_prompt_path = "mu/algorithms/selective_amnesia/data/fim_prompts_sample.txt"
    )
    algorithm.run()
```

**Run the script**


```bash
WANDB_MODE=offline python my_trainer.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

## Notes

1. Ensure all dependencies are installed as per the environment file.
2. The training process generates logs in the `logs/` directory for easy monitoring.
3. Use appropriate CUDA devices for optimal performance during training.
4. Regularly verify dataset and model configurations to avoid errors during execution.
---

