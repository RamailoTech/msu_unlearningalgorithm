Train your model by using ScissorHands Algorithm. Import pre defined config classes or create your own object.
Refer the config docs for details about the parameters that you can use.

To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

### Use pre defined config
```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import scissorhands_train_mu
algorithm = ScissorHandsAlgorithm(scissorhands_train_mu)
algorithm.run()
```

### Modify some train parameters in pre defined config class.
View the config docs to see a list of available parameters.
```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import (
    scissorhands_train_mu,
)

algorithm = ScissorHandsAlgorithm(
    scissorhands_train_mu,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
)
algorithm.run()
```

### Create your own config object
```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import (
    ScissorHandsConfig,
)

myconfig = ScissorHandsConfig()
myconfig.ckpt_path = "models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

algorithm = ScissorHandsAlgorithm(myconfig)
algorithm.run()
```

### Override the Config class itself.
```python
from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.algorithms.scissorhands.configs import (
    ScissorHandsConfig,
)


class MyNewConfigClass(ScissorHandsConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()


new_config_object = MyNewConfigClass()
algorithm = ScissorHandsAlgorithm(new_config_object)
algorithm.run()
```