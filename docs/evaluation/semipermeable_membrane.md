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

evaluator = SemipermeableMembraneEvaluator(
    semipermeable_membrane_eval_config,
    ckpt_path="outputs/semipermeable_membrane/finetuned_models/semipermeable_membrane_Abstractionism_last.safetensors",
    spm_path = ["outputs/semipermeable_membrane/finetuned_models/semipermeable_membrane_Abstractionism_last.safetensors"],
    classifier_ckpt_path = "models/classifier_ckpt_path/style50_cls.pth",
    reference_dir= "data/quick-canvas-dataset/sample/"
)
evaluator.run()
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

- base_model : Path to the pre-trained base model used for image generation.  
   - *Type:* `str`  
   - *Example:* `"path/to/base/model.pth"`  

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
- theme : Specifies the theme for the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`  

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

- eval_output_dir : Directory where evaluation metrics and results will be stored.  
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
- multiprocessing : Enables multiprocessing for faster evaluation.  
   - *Type:* `bool`  
   - *Example:* `false`  

- seed_list :  List of random seeds for multiple evaluation trials.  
   - *Type:* `list`  
   - *Example:* `["188"]`  