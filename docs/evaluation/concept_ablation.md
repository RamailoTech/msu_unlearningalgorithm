#### Concept ablation Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Concept ablation algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.

#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file. Work within the same environment used to perform unlearning for evaluation as well.


**Before running evaluation, download the classifier ckpt from [here](https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61).**

**Example Code**

```python
from mu.algorithms.concept_ablation import ConceptAblationEvaluator
from mu.algorithms.concept_ablation.configs import (
    concept_ablation_evaluation_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.fid import fid_score

evaluator = ConceptAblationEvaluator(
    concept_ablation_evaluation_config,
    ckpt_path="outputs/concept_ablation/checkpoints/last.ckpt",
)
generated_images_path = evaluator.generate_images()

reference_image_dir = "data/quick-canvas-dataset/sample"

accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                          classifier_ckpt_path = "models/classifier_ckpt_path/style50_cls.pth",
                          reference_dir=reference_image_dir,
                          forget_theme="Bricks",
                          )
print(accuracy['acc'])
print(accuracy['loss'])

fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=reference_image_dir )

print(fid)
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```

**How It Works** 
* Default Values: The script first loads default values from the evluation config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the evaluation. 


