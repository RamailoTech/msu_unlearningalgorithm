
### Example usage to Run Defense for compvis 

To test the below code snippet, you can create a file, copy the below code in eg, `mu_defense.py` and execute it with `python mu_defense.py` or use `WANDB_MODE=offline python mu_defense.py` for offline mode.

### Run with default config

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config

    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
 
```

### Modify some train parameters in pre defined config class.

View the config descriptions to see a list of available parameters.

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        compvis_ckpt_path = "outputs/erase_diff/erase_diff_Abstractionism_model.pth",
        attack_step = 2,
        backend = "compvis",
        attack_method = "fast_at",
        warmup_iter = 1,
        iterations = 2,
        model_config_path = erase_diff_train_mu.model_config_path

    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
 
```

**Note: import the model_config_path from the relevant algorithm's configuration module in the mu package**


### Example usage to Run defense for diffusers


### Run with default config

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config
    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
```


### Modify some train parameters in pre defined config class.

View the config descriptions to see a list of available parameters.

```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        diffusers_model_name_or_path = "outputs/forget_me_not/finetuned_models/Abstractionism",
        attack_step = 2,
        backend = "diffusers",
        attack_method = "fast_at",
        warmup_iter = 1,
        iterations = 2
    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
```


