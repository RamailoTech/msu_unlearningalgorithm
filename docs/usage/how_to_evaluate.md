**Evaluation using existing Unlearn canvas dataset:**

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
                          seed_list = ["188"] )
print(accuracy['acc'])
print(accuracy['loss'])

reference_image_dir = "data/quick-canvas-dataset/sample"
fid, _ = fid_score(generated_image_dir=generated_images_path,
                reference_image_dir=reference_image_dir )

print(fid)
```



**Link to our example usage notebooks**

We have kept full implementation of machine unlearning, attack and defense in these notebooks.

1. **Erase-diff (compvis model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_erase_diff.ipynb

2. **forget-me-not (Diffuser model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_forget_me_not.ipynb
