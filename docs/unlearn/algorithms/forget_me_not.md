
**Run Train**
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Train a Text Inversion using quick canvas dataset**

Before finetuning the model you need to generate safetensors.

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
    steps=10,
    template_name = "Abstractionism", #concept to erase
    dataset_type = "unlearncanvas" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/forget_me_not/finetuned_models" #output dir to save finetuned models
)
algorithm.run(train_type="train_ti")
```

**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python my_trainer_ti.py
```

2. **Perform Unlearning using quick canvas dataset**

Before running the `train_attn` script, update the `ti_weights_path` parameter in the configuration file to point to the output generated from the Text Inversion (train_ti.py) stage

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
    ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors",
    template_name = "Abstractionism", #concept to erase
    dataset_type = "unlearncanvas" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/forget_me_not/finetuned_models" #output dir to save finetuned models
)
algorithm.run(train_type="train_attn")
```


**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python my_trainer_attn.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 


**This method involves two stages:**

1. **Train a Text Inversion**: The first stage involves training a Text Inversion. Refer to the script [`train_ti.py`](mu/algorithms/forget_me_not/scripts/train_ti.py) for details and implementation. It uses `train_ti_config.yaml` as config file.

2. **Perform Unlearning**: The second stage uses the outputs from the first stage to perform unlearning. Refer to the script [`train_attn.py`](mu/algorithms/forget_me_not/scripts/train_attn.py) for details and implementation. It uses `train_attn_config.yaml` as config file.



