
# Concept Ablation Algorithm for Machine Unlearning

This repository provides an implementation of the Concept Ablation algorithm for machine unlearning in Stable Diffusion models. The Concept Ablation algorithm enables the removal of specific concepts or styles from a pre-trained model without the need for retraining from scratch.


### Example usage for quick canvas dataset

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


### Example usage for i2p dataset

Add the following code snippet to a python script `trainer.py`. Run the script using `python trainer.py`.

```python
from mu.algorithms.concept_ablation.algorithm import (
    ConceptAblationAlgorithm,
)
from mu.algorithms.concept_ablation.configs import (
    concept_ablation_train_i2p,
    ConceptAblationConfig,
)

if __name__ == "__main__":

    concept_ablation_train_i2p.lightning.trainer.max_steps = 5

    algorithm = ConceptAblationAlgorithm(
        concept_ablation_train_i2p,
        config_path="mu/algorithms/concept_ablation/configs/train_config.yaml",
        raw_dataset_dir = "data/i2p-dataset/sample",
        ckpt_path="models/compvis/style50/compvis.ckpt",
        prompts="mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt",
        output_dir="/opt/dlami/nvme/outputs",
        template_name = "self-harm", #concept to erase
        dataset_type = "i2p" ,
        use_sample = True, #train on sample dataset
        # devices="1",
    )
    algorithm.run()
```


**Use your own dataset for unlearning**

**Step-1: Generate your own dataset**

```bash
generate_images_for_prompts --model_path models/diffuser/style50 --csv_path data/prompts/generic_data.csv
```

Note:

* generate_images_for_prompts: This command invokes the image generation script. It uses a diffusion model to generate images based on textual prompts.

* --model_path: Specifies the path to the diffusion model to be used for image generation. In this example, the model is located at models/diffuser/style50.

* --csv_path: Provides the path to a CSV file containing the prompts. Each prompt in this CSV will be used to generate an image, allowing you to build a dataset tailored to your needs.


**Step-2: Train on your own dataset**

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
    raw_dataset_dir="data/generic", #replace with your own generated path
    prompt_path = "data/generic/prompts/generic_data.csv",
    dataset_type = "generic", #to use you own dataset use dataset type as generic
    template_name = "self-harm",
    output_dir="outputs/uce",
)
algorithm.run()
```

## Notes

1. Ensure all dependencies are installed as per the environment file.
2. The training process generates logs in the `logs/` directory for easy monitoring.
3. Use appropriate CUDA devices for optimal performance during training.
4. Regularly verify dataset and model configurations to avoid errors during execution.
---


## Directory Structure

- `algorithm.py`: Core implementation of the Concept Ablation Algorithm.
- `configs/`: Configuration files for training and generation.
- `data_handler.py`: Data handling and preprocessing.
- `scripts/train.py`: Script to train the Concept Ablation Algorithm.
- `callbacks/`: Custom callbacks for logging and monitoring training.
- `utils.py`: Utility functions.



