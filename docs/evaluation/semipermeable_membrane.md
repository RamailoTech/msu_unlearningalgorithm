#### Semipermeable membrane Evaluation Framework

This section provides instructions for running the **evaluation framework** for the Semipermeable membrane algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/semipermeable_membrane/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


### **Basic Command to Run Evaluation:**

**Before running evaluation, download the classifier ckpt from here:**

https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61 

Add the following code to `evaluate.py`

```python
from mu.algorithms.semipermeable_membrane import SemipermeableMembraneEvaluator
from mu.algorithms.semipermeable_membrane.configs import (
    semipermeable_membrane_eval_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.fid import fid_score


evaluator = SemipermeableMembraneEvaluator(
    semipermeable_membrane_eval_config,
    spm_path = ["outputs/semipermiable/semipermeable_membrane_Abstractionism_last.safetensors"],
)
generated_images_path = evaluator.generate_images()

reference_image_dir = "/home/ubuntu/Projects/Palistha/testing/data/quick-canvas-dataset/sample"

accuracy = accuracy_score(gen_image_dir=generated_images_path,
                          dataset_type = "unlearncanvas",
                          classifier_ckpt_path = "/home/ubuntu/Projects/models/classifier_ckpt_path/style50_cls.pth",
                          reference_dir=reference_image_dir,
                          forget_theme="Bricks",
                          seed_list = ["188"] )
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


#### **Description of parameters in evaluation_config.yaml**

The `evaluation_config.yaml` file contains the necessary parameters for running the Semipermeable membrane evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration Parameters:**
- spm_path: paths to finetuned model checkpoint.
   - *Type:* `list`
   - *Example:* `outputs/semipermeable_membrane/finetuned_models/semipermeable_membrane_Abstractionism_last.safetensors`


- precision : Specifies the numerical precision for model computation.  
   - *Type:* `str`  
   - *Options:* `"fp32"`, `"fp16"`, `"bf16"`  
   - *Example:* `"fp32"`  

- spm_multiplier:  Specifies the multiplier for Semipermeable Membrane (SPM) model.  
   - *Type:* `float`  
   - *Example:* `1.0`  

- v2 : Specifies whether to use version 2.x of the model.  
   - *Type:* `bool`  
   - *Example:* `false`  

- matching_metric : Metric used for evaluating the similarity between generated prompts and erased concepts.  
   - *Type:* `str`  
   - *Options:* `"clipcos"`, `"clipcos_tokenuni"`, `"tokenuni"`  
   - *Example:* `"clipcos_tokenuni"`  

- model_config : Path to the model configuration YAML file.  
   - *Type:* `str`  
   - *Example:* `"mu/algorithms/semipermeable_membrane/config"`  

- model_ckpt_path: Path to pretrained Stable Diffusion model.
   - *Type*: `str`
   - *Example*: `models/diffuser/style50`

---

### **Sampling Parameters:**

- seed : Random seed for reproducibility of the evaluation process.  
   - *Type:* `int`  
   - *Example:* `188`  

- devices : Specifies the CUDA devices for running the model.  
   - *Type:* `str` (Comma-separated for multiple devices)  
   - *Example:* `"0"`  

- task : Specifies the task type for the evaluation process.  
   - *Type:* `str`  
   - *Options:* `"class"`, `"style"`  
   - *Example:* `"class"`  

---

### **Output Parameters:**
- sampler_output_dir : Directory where generated images will be saved during the sampling process.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/semipermeable_membrane/"`  

---

### **Dataset and Classification Parameters:**
- reference_dir : Path to the reference dataset used for evaluation and comparison.  
   - *Type:* `str`  
   - *Example:* `"msu_unlearningalgorithm/data/quick-canvas-dataset/sample/"`  

- classification_model : Specifies the classification model used for the evaluation.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`  

- forget_theme : Specifies the theme to be forgotten during the unlearning process.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`  

---

### **Performance Parameters:**

- seed_list :  List of random seeds for multiple evaluation trials.  
   - *Type:* `list`  
   - *Example:* `["188"]`  

- use_sample: If you want to just run on sample dataset then set it as True. By default it is True.
   - *Type:* `bool`  
   - *Example:* `True`