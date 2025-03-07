## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Using quick canvas dataset**

```python
from mu.algorithms.unified_concept_editing.algorithm import (
    UnifiedConceptEditingAlgorithm,
)
from mu.algorithms.unified_concept_editing.configs import (
    unified_concept_editing_train_mu,
)

algorithm = UnifiedConceptEditingAlgorithm(
    unified_concept_editing_train_mu,
    ckpt_path="models/diffuser/style50/",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
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
