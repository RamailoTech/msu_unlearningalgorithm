Train your model by using Semi Permeable Membrane Algorithm. Import pre defined config classes or create your own object.
Refer the config docs for details about the parameters that you can use.

To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

### Use Pre defined config class
```python
from mu.algorithms.semipermeable_membrane.algorithm import (
    SemipermeableMembraneAlgorithm,
)
from mu.algorithms.semipermeable_membrane.configs import semipermiable_membrane_train_mu
algorithm = SemipermeableMembraneAlgorithm(semipermiable_membrane_train_mu)
algorithm.run()
```

### Modify some parameters in pre defined config class
```python
from mu.algorithms.semipermeable_membrane.algorithm import (
    SemipermeableMembraneAlgorithm,
)
from mu.algorithms.semipermeable_membrane.configs import (
    semipermiable_membrane_train_mu,
)

algorithm = SemipermeableMembraneAlgorithm(
    semipermiable_membrane_train_mu,
    output_dir="/opt/dlami/nvme/outputs",
)
algorithm.run()
```


### Create your own config object
```python
from mu.algorithms.semipermeable_membrane.algorithm import SemipermeableMembraneAlgorithm
from mu.algorithms.semipermeable_membrane.configs import (
    SemipermeableMembraneConfig,
)

myconfig = SemipermeableMembraneConfig()
myconfig.output_dir = (
    "/opt/dlami/nvme/outputs"
)

algorithm = SemipermeableMembraneAlgorithm(myconfig)
algorithm.run()

```

### Override the Config class itself.
```python
from mu.algorithms.semipermeable_membrane.algorithm import SemipermeableMembraneAlgorithm
from mu.algorithms.semipermeable_membrane.configs import (
    SemipermeableMembraneConfig,
)


class MyNewConfigClass(SemipermeableMembraneConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()

new_config_object = MyNewConfigClass()
algorithm = SemipermeableMembraneAlgorithm(new_config_object)
algorithm.run()

```