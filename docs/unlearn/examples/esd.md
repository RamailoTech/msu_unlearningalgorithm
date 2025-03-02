Train your model by using Esd Algorithm. Import pre defined config classes or create your own object.
Refer the config docs for details about the parameters that you can use.

To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

### Use pre defined config

```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import esd_train_mu

algorithm = ESDAlgorithm(esd_train_mu)
algorithm.run()
```

### Modify some train parameters in pre defined config class.
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    esd_train_mu,
)

algorithm = ESDAlgorithm(
    esd_train_mu,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
)
algorithm.run()
```

### Create your own config object
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    ESDConfig,
)

myconfig = ESDConfig()
myconfig.ckpt_path = "models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

algorithm = ESDAlgorithm(myconfig)
algorithm.run()
```

### Override the Config class itself.
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    ESDConfig,
)


class MyNewConfigClass(ESDConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()


new_config_object = MyNewConfigClass()
algorithm = ESDAlgorithm(new_config_object)
algorithm.run()
```
