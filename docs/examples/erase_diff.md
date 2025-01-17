Train your model by using Erase Diff Algorithm
```
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.helpers import load_config


config_path= "train_config.yaml"
config = load_config(config_path)
algorithm = EraseDiffAlgorithm(config)
algorithm.run()
```