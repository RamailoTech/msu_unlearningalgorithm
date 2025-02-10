```python
from mu_defense.algorithms.adv_unlearn.algorithm import AdvUnlearnAlgorithm
from mu_defense.algorithms.adv_unlearn.configs import adv_unlearn_config
from mu.algorithms.erase_diff.configs import erase_diff_train_mu


def mu_defense():

    mu_defense = AdvUnlearnAlgorithm(
        config=adv_unlearn_config,
        compvis_ckpt_path = "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/erase_diff/erase_diff_Abstractionism_model.pth",
        # diffusers_model_name_or_path = "/home/ubuntu/Projects/dipesh/unlearn_diff/outputs/forget_me_not/finetuned_models/Abstractionism",
        attack_step = 2,
        backend = "compvis",
        attack_method = "fast_at",
        model_config_path = erase_diff_train_mu.model_config_path
        

    )
    mu_defense.run()

if __name__ == "__main__":
    mu_defense()
 
```