
**Run Train using quick canvas dataset**

Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Example Code**
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    esd_train_mu,
)

algorithm = ESDAlgorithm(
    esd_train_mu,
    ckpt_path="machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    template_name = "Abstractionism", #concept to erase
    dataset_type = "unlearncanvas" ,
    template = "class",
    use_sample = True, #train on sample dataset
    output_dir = "outputs/esd/finetuned_models" #output dir to save finetuned models
    
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

