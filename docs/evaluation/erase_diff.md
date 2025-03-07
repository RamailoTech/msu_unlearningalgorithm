#### Erase Diff Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Erase Diff algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/erase_diff/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


### **Basic Command to Run Evaluation:**

**Before running evaluation, download the classifier ckpt from [here](https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61).**



Add the following code to `evaluate.py`.


```python

from mu.algorithms.erase_diff import EraseDiffEvaluator
from mu.algorithms.erase_diff.configs import (
    erase_diff_evaluation_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.fid import fid_score


evaluator = EraseDiffEvaluator(
    erase_diff_evaluation_config,
    ckpt_path="outputs/erase_diff/finetuned_models/erase_diff_self-harm_model.pth",
)
generated_images_path = evaluator.generate_images()

accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                        classifier_ckpt_path = "/home/ubuntu/Projects/models/classifier_ckpt_path/style50_cls.pth",
                          forget_theme="Bricks",
                          )
print(accuracy['acc'])
print(accuracy['loss'])

reference_image_dir = "data/quick-canvas-dataset/sample"
fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=reference_image_dir )

print(fid)
```

**Run the script**

```bash
python evaluate.py
```


