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
from evaluation.metrics.clip import clip_score
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


fid, _ = fid_score(generated_image_dir=generated_images_path) #Defaults to the COCO dataset if reference_image_dir is not provided."
print(fid)

clip_score = clip_score() #Defaults to the COCO dataset if image path is not provided."
print(clip_score)

#optionally provide your own dataset path
fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=reference_image_dir )

print(fid)
```

#### **Classifiers for evaluation**

**Nudenet classifier**
Requirements for Testing:

To test this function, the following directory structure and files must be present in the root directory:

* imgs/ folder - Must be located inside the root directory (data/i2p/imgs/)

* prompts.csv file - Must be present in the root directory (data/i2p/prompts.csv)

```python
from evaluation.metrics.nudenet import process_nudity_classification

process_nudity_classification()

#Optionally, to run with custom parameters
process_nudity_classification(root='data/i2p', mode='check_none_nude', threshold=0.5)
```

**Q16 classifier**

```python
import torch

from evaluation.metrics.q16 import q16_binary_classifier, harm_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_FILE = 'mu_attack/tasks/utils/metrics/q16/prompts.p' #replace with your own path
IMAGE_TO_TEST = 'data/i2p/imgs/image_test.png' #replace with your own test image path

clip_model, classifier = q16_binary_classifier(device=DEVICE, prompt_path=PROMPT_FILE)
label, scores = harm_eval(clip_model, classifier, IMAGE_TO_TEST, DEVICE)
label_text = "Benign" if label == 0 else "Malicious"
print(f"\nFinal Label: {label} ({label_text})")
print(f"Confidence Scores:")
print(f"  - Benign:    {scores[0].item():.4f}")
print(f"  - Malicious: {scores[1].item():.4f}")
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python evaluate.py
```

**How It Works** 
* Default Values: The script first loads default values from the evluation config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the evaluation. 


