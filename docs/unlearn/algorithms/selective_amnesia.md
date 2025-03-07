
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

