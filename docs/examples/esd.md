### Use pre defined config

```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import esd_train_config_quick_canvas

algorithm = ESDAlgorithm(esd_train_config_quick_canvas)
algorithm.run()
```

### Modify some train parameters in pre defined config class.
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    esd_train_config_quick_canvas,
)

algorithm = ESDAlgorithm(
    esd_train_config_quick_canvas,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
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
myconfig.ckpt_path = "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
)

algorithm = ESDAlgorithm(myconfig)
algorithm.run()
```

### Override the Config class itself.
```python
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.algorithms.esd.configs import (
    esd_train_config_quick_canvas,
)


class MyNewConfigClass(ESDAlgorithm):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()


new_config_object = MyNewConfigClass()
algorithm = ESDAlgorithm(new_config_object)
algorithm.run()
```
