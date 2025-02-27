To perform training using selective amnesia. You'll need to download the full fisher file first. 
Download, it at `mu/algorithms/selective_amnesia/data` folder.

**Use the following command**

```bash
wget https://huggingface.co/ajrheng/selective-amnesia/resolve/main/full_fisher_dict.pkl
```

### Use pre defined config
```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    selective_amnesia_config_quick_canvas,
)

algorithm = SelectiveAmnesiaAlgorithm(
    selective_amnesia_config_quick_canvas
)
algorithm.run()

```

### Modify some train parameters in pre defined config class.
View the config docs to see a list of available parameters.


```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    selective_amnesia_config_quick_canvas,
)

algorithm = SelectiveAmnesiaAlgorithm(
    selective_amnesia_config_quick_canvas,
    ckpt_path="models/compvis/style50/compvis.ckpt",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
)
algorithm.run()

```

### Create your own config object
```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    SelectiveAmnesiaConfig,
)


myconfig = SelectiveAmnesiaConfig()
myconfig.ckpt_path = "models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

algorithm = SelectiveAmnesiaAlgorithm(myconfig)
algorithm.run()
```

### Override the Config class itself.

```python
from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.algorithms.selective_amnesia.configs import (
    SelectiveAmnesiaConfig,
)


class MyNewConfigClass(SelectiveAmnesiaAlgorithm):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()


new_config_object = MyNewConfigClass()
algorithm = SelectiveAmnesiaAlgorithm(new_config_object)
algorithm.run()
```
