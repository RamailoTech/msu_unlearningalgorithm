#### Saliency Unlearning Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Saliency Unlearning algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/saliency_unlearning/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


### **Basic Command to Run Evaluation:**

**Before running evaluation, download the classifier ckpt from [here](https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61).**


Add the following code to `evaluate.py`

```python

from mu.algorithms.saliency_unlearning import SaliencyUnlearningEvaluator
from mu.algorithms.saliency_unlearning.configs import (
    saliency_unlearning_evaluation_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.fid import fid_score

evaluator = SaliencyUnlearningEvaluator(
    saliency_unlearning_evaluation_config,
    ckpt_path="outputs/saliency_unlearning/saliency_unlearning_Abstractionism_model.pth",
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

**Run the script**

```bash
python evaluate.py
```



