#### Semipermeable membrane Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Semipermeable membrane algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/semipermeable_membrane/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


### **Basic Command to Run Evaluation:**

**Before running evaluation, download the classifier ckpt from [here](https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61).**


Add the following code to `evaluate.py`

```python
from mu.algorithms.semipermeable_membrane import SemipermeableMembraneEvaluator
from mu.algorithms.semipermeable_membrane.configs import (
    semipermeable_membrane_eval_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.clip import clip_score
from evaluation.metrics.fid import fid_score


evaluator = SemipermeableMembraneEvaluator(
    semipermeable_membrane_eval_config,
    spm_path = ["outputs/semipermiable/semipermeable_membrane_Abstractionism_last.safetensors"],
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

**Run the script**

```bash
python evaluate.py
```


