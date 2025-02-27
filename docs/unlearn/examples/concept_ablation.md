
Train your model by using Concept Ablation Algorithm. Import pre defined config classes or create your own object.
Refer the config docs for details about the parameters that you can use.


To test the below code snippet, you can create a file, copy the below code in eg, `my_trainer.py`
and execute it with `python my_trainer.py` or use `WANDB_MODE=offline python my_trainer.py` for offline mode.


### Use pre defined config

```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    concept_ablation_train_mu,
    ConceptAblationConfig,
)

if __name__ == "__main__":

    concept_ablation_train_mu.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_mu
    )
    algorithm.run()
```


### Modify some train parameters in pre defined config class.
View the config docs to see a list of available parameters.

```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    concept_ablation_train_mu,
    ConceptAblationConfig,
)

if __name__ == "__main__":

    concept_ablation_train_mu.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_mu,
        config_path="mu/algorithms/concept_ablation/configs/train_config.yaml",
        ckpt_path="machine_unlearning/models/compvis/style50/compvis.ckpt",
        prompts="mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt",
        output_dir="/opt/dlami/nvme/outputs",
        # devices="1",
    )
    algorithm.run()
```


### Create your own config object


```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    ConceptAblationConfig,
)

myconfig = ConceptAblationConfig()
myconfig.ckpt_path = "machine_unlearning/models/compvis/style50/compvis.ckpt"
myconfig.raw_dataset_dir = (
    "/home/ubuntu/Projects/balaram/packaging/data/quick-canvas-dataset/sample"
)
algorithm = ConceptAblationConfig(myconfig)
algorithm.run()
```

### Override the Config class itself.

```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    ConceptAblationConfig,
)


class MyNewConfigClass(ConceptAblationConfig):
    def __init__(self, *args, **kwargs):
        self.new_parameter = kwargs.get("new_parameter")
        super().__init__()

new_config_object = MyNewConfigClass()
algorithm = ConceptAblationAlgorithm(new_config_object)
algorithm.run()

```