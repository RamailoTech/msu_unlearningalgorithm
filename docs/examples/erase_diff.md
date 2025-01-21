Train your model by using Erase Diff Algorithm. Import pre defined config classes or create your own object.

To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

### Use pre defined config
```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_config_quick_canvas

algorithm = EraseDiffAlgorithm(erase_diff_train_config_quick_canvas)
algorithm.run()
```

### Modify some train parameters in pre defined config class.
```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import (
    erase_diff_train_config_quick_canvas,
)

algorithm = EraseDiffAlgorithm(
    erase_diff_train_config_quick_canvas,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ),
)
algorithm.run()
```

### Create your own config object
```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import (
    EraseDiffConfig,
)

myconfig = EraseDiffConfig()
myconfig.ckpt_path = "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
)

algorithm = EraseDiffAlgorithm(myconfig)
algorithm.run()

```

### Override the Config class itself.

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import (
    erase_diff_train_config_quick_canvas,
)


class MyNewConfigClass(EraseDiffAlgorithm):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()

new_config_object = MyNewConfigClass()
algorithm = EraseDiffAlgorithm(new_config_object)
algorithm.run()

```