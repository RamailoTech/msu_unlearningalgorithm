Train your model by using Unified Concept Editing Algorithm. Import pre defined config classes or create your own object.
Refer the config docs for details about the parameters that you can use.

To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.

### Use Pre defined config class
```python
from mu.algorithms.unified_concept_editing.algorithm import (
    UnifiedConceptEditingAlgorithm,
)
from mu.algorithms.unified_concept_editing.configs import unified_concept_editing_train_mu
algorithm = UnifiedConceptEditingAlgorithm(unified_concept_editing_train_mu)
algorithm.run()
```

### Modify some parameters in pre defined config class
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


### Create your own config object
```python
from mu.algorithms.unified_concept_editing.algorithm import UnifiedConceptEditingAlgorithm
from mu.algorithms.unified_concept_editing.configs import (
    UnifiedConceptEditingConfig,
)

myconfig = UnifiedConceptEditingConfig()
myconfig.ckpt_path = "models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

algorithm = UnifiedConceptEditingAlgorithm(myconfig)
algorithm.run()

```

### Override the Config class itself.
```python
from mu.algorithms.unified_concept_editing.algorithm import UnifiedConceptEditingAlgorithm
from mu.algorithms.unified_concept_editing.configs import (
    UnifiedConceptEditingConfig,
)
UnifiedConceptEditingAlgorithm

class MyNewConfigClass(UnifiedConceptEditingConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()

new_config_object = MyNewConfigClass()
algorithm = UnifiedConceptEditingAlgorithm(new_config_object)
algorithm.run()

```