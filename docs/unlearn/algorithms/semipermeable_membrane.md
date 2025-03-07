## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  


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
