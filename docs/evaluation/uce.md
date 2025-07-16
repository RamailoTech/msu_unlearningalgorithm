#### unified_concept_editing Evaluation Framework

This section provides instructions for running the **evaluation framework** for the unified_concept_editing algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/unified_concept_editing/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


**Before running evaluation, download the classifier ckpt from [here](https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61).**



Then, Add the following code to `evaluate.py`

```python
from mu.algorithms.unified_concept_editing import UnifiedConceptEditingEvaluator
from mu.algorithms.unified_concept_editing.configs import (
    uce_evaluation_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.fid import fid_score


# reference_image_dir = "data/generic"
evaluator = UnifiedConceptEditingEvaluator(
    uce_evaluation_config,
    ckpt_path="outputs/uce/uce_Abstractionism_model",
)
# model = evaluator.load_model()
generated_images_path = evaluator.generate_images()

reference_image_dir = "data/quick-canvas-dataset/sample"

accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                          classifier_ckpt_path = "/home/ubuntu/Projects/models/classifier_ckpt_path/style50_cls.pth",
                          reference_dir=reference_image_dir,
                          forget_theme="Bricks",
                          )
print(accuracy['acc'])
print(accuracy['loss'])

fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=reference_image_dir )

print(fid)
```


#### **Classifiers for evaluation**

**Nudenet classifier**

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

**Running in Offline Mode:**

```bash
WANDB_MODE=offline python evaluate.py
```

