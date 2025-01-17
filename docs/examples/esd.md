Train your model by using ESD Algorithm
```
from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.helpers import load_config


config = load_config('train_config.yaml')
algorithm = ESDAlgorithm(config)
algorithm.run()
```