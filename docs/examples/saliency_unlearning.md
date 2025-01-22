Train your model by using Saliency Unlearning Algorithm. Import pre defined config classes or create your own object.
Refer the config docs for details about the parameters that you can use.

To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

**Note**:
- You must first run the generate_mask script, before running the train script below. Refer algorithm's Usage section for this.

### Use Pre defined config class
```python
from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_train_config_quick_canvas
algorithm = SaliencyUnlearnAlgorithm(saliency_unlearning_train_config_quick_canvas)
algorithm.run()
```

### Modify some parameters in pre defined config class
```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearnAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_config_quick_canvas,
)

algorithm = SaliencyUnlearnAlgorithm(
    saliency_unlearning_train_config_quick_canvas,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
)
algorithm.run()
```


### Create your own config object
```python
from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
from mu.algorithms.saliency_unlearning.configs import (
    SaliencyUnlearningConfig,
)

myconfig = SaliencyUnlearningConfig()
myconfig.ckpt_path = "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
)

algorithm = SaliencyUnlearnAlgorithm(myconfig)
algorithm.run()

```

### Override the Config class itself.
```python
from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
from mu.algorithms.saliency_unlearning.configs import (
    SaliencyUnlearningConfig,
)

class MyNewConfigClass(SaliencyUnlearningConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()

new_config_object = MyNewConfigClass()
algorithm = SaliencyUnlearnAlgorithm(new_config_object)
algorithm.run()

```

