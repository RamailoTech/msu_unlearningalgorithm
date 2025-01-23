Train your model by using Saliency Unlearning Algorithm. Import pre defined config classes or create your own object.
Refer to the config docs for details about the parameters that you can use.

To test the below code snippet, you can create a file, copy the below code in, e.g., `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

### Use Pre-defined config class
```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearningAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import saliency_unlearning_train_mu
algorithm = SaliencyUnlearningAlgorithm(saliency_unlearning_train_mu)
algorithm.run()
```

### Modify some parameters in pre-defined config class
```python
from mu.algorithms.saliency_unlearning.algorithm import (
    SaliencyUnlearningAlgorithm,
)
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_train_mu,
)

algorithm = SaliencyUnlearningAlgorithm(
    saliency_unlearning_train_mu,
    output_dir="/opt/dlami/nvme/outputs",
)
algorithm.run()
```

### Create your own config object
```python
from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearningAlgorithm
from mu.algorithms.saliency_unlearning.configs import (
    SaliencyUnlearningConfig,
)

myconfig = SaliencyUnlearningConfig()
myconfig.output_dir = (
    "/opt/dlami/nvme/outputs"
)

algorithm = SaliencyUnlearningAlgorithm(myconfig)
algorithm.run()
```

### Override the Config class itself
```python
from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearningAlgorithm
from mu.algorithms.saliency_unlearning.configs import (
    SaliencyUnlearningConfig,
)

class MyNewConfigClass(SaliencyUnlearningConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()

new_config_object = MyNewConfigClass()
algorithm = SaliencyUnlearningAlgorithm(new_config_object)
algorithm.run()
```
