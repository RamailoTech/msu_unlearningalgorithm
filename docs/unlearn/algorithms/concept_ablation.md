
**Example Usage of existing unlearn algorithms**

Add the following code snippet to a python script `trainer.py`. Run the script using `python trainer.py`.

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
        template_name = "Abstractionism", #concept to erase
        dataset_type = "unlearncanvas" ,
        use_sample = True, #train on sample dataset
        # devices="1",
    )
    algorithm.run()
```


**Notes**

1. Ensure all dependencies are installed as per the environment file.
2. The training process generates logs in the `logs/` directory for easy monitoring.
3. Use appropriate CUDA devices for optimal performance during training.
4. Regularly verify dataset and model configurations to avoid errors during execution.
---
