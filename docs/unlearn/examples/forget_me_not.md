You can modify the parameters, when using config class itself.
View the config docs to see a list of available parameters that you can use.


### Train a text inversion (train_ti)

**Using default config**


```python

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_ti_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_ti_mu
)
algorithm.run(train_type="train_ti")
```

**Modify some parameters in pre-defined config class**

```python

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_ti_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_ti_mu,
    ckpt_path="models/diffuser/style50",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ), 
    steps=10
)
algorithm.run(train_type="train_ti")
```


**Create your own config class**

```python

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    ForgetMeNotTiConfig,
)

myconfig = ForgetMeNotTiConfig()
myconfig.ckpt_path = "models/diffuser/style50"
myconfig.steps = 1

algorithm = ForgetMeNotAlgorithm(
    myconfig
)
algorithm.run(train_type="train_ti")
```



### Perform unlearning by using train attn.
Before running the `train_attn` script, update the `ti_weights_path` parameter in the configuration file to point to the output generated from the Text Inversion (train_ti.py) stage

**Using default config**

```python
from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_attn_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_attn_mu,
)
algorithm.run(train_type="train_attn")
```

**Modify some parameters in pre-defined config class**

```python
from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_attn_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_attn_mu,
    ckpt_path="models/diffuser/style50",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    steps=10,
    ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors"
)
algorithm.run(train_type="train_attn")
```

**Create your own config class**


```python
from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    ForgetMeNotAttnConfig,
)

myconfig = ForgetMeNotAttnConfig()
myconfig.ckpt_path = "models/diffuser/style50"
myconfig.steps = 1

algorithm = ForgetMeNotAlgorithm(
    myconfig
)
algorithm.run(train_type="train_attn")
```
