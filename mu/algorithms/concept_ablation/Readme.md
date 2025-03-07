
# Concept Ablation Algorithm for Machine Unlearning

This repository provides an implementation of the Concept Ablation algorithm for machine unlearning in Stable Diffusion models. The Concept Ablation algorithm enables the removal of specific concepts or styles from a pre-trained model without the need for retraining from scratch.

---

## Installation

#### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

Before installing the unlearn_diff package, follow these steps to set up your environment correctly. These instructions ensure compatibility with the required dependencies, including Python, PyTorch, and ONNX Runtime.


**Step-by-Step Setup:**

Step 1. Create a Conda Environment Create a new Conda environment named myenv with Python 3.8.5:

```bash
conda create -n myenv python=3.8.5
```

Step 2. Activate the Environment Activate the environment to work within it:

```bash
conda activate myenv
```

Step 3. Install Core Dependencies Install PyTorch, torchvision, CUDA Toolkit, and ONNX Runtime with specific versions:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 onnxruntime==1.16.3 -c pytorch -c conda-forge
```

Step 4. Install our unlearn_diff Package using pip:

```bash
pip install unlearn_diff
```

Step 5. Install Additional Git Dependencies:

 After installing unlearn_diff, install the following Git-based dependencies in the same Conda environment to ensure full functionality:

```bash
pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```

```bash
pip install git+https://github.com/openai/CLIP.git@main#egg=clip
```

```bash
pip install git+https://github.com/crowsonkb/k-diffusion.git
```

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

```bash
pip install git+https://github.com/Phoveran/fastargs.git@main#egg=fastargs
```

```bash
pip install git+https://github.com/boomb0om/text2image-benchmark
```



### Downloading data and models.
After you install the package, you can use the following commands to download.

1. **Dataset**:
  - **i2p**:
    - **Sample**:
     ```
     download_data sample i2p
     ```
    - **Full**:
     ```
     download_data full i2p
     ```
  - **quick_canvas**:
    - **Sample**:
     ```
     download_data sample quick_canvas
     ```
    - **Full**:
     ```
     download_data full quick_canvas
     ```

2. **Model**:
  - **compvis**:
    ```
    download_model compvis
    ```
  - **diffuser**:
    ```
    download_model diffuser
    ```
3. **Download best.onnx model**

 ```
 download_best_onnx
 ```

**Verify the Downloaded Files**

After downloading, verify that the datasets have been correctly extracted:
```bash
ls -lh ./data/i2p-dataset/sample/
ls -lh ./data/quick-canvas-dataset/sample/
```
---

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



## Configuration File (`train_config.yaml`)

### Training Parameters

* **seed:** Random seed for reproducibility.
    * Type: int
    * Example: 23

* **scale_lr:** Whether to scale the base learning rate.
    * Type: bool
    * Example: True

* **caption_target:** Target style to remove.
    * Type: str
    * Example: "Abstractionism Style"

* **regularization:** Adds regularization loss during training.
    * Type: bool
    * Example: True

* **n_samples:** Number of batch sizes for image generation.
    * Type: int
    * Example: 10

* **train_size:** Number of generated images for training.
    * Type: int
    * Example: 1000

* **base_lr:** Learning rate for the optimizer.
    * Type: float
    * Example: 2.0e-06

### Model Configuration

* **model_config_path:** Path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/model_config.yaml"

* **ckpt_path:** Path to the Stable Diffusion model checkpoint.
    * Type: str
    * Example: "/path/to/compvis.ckpt"

### Dataset Directories

* **raw_dataset_dir:** Directory containing the raw dataset categorized by themes or classes.
    * Type: str
    * Example: "/path/to/raw_dataset"

* **processed_dataset_dir:** Directory to save the processed dataset.
    * Type: str
    * Example: "/path/to/processed_dataset"

* **dataset_type:** Specifies the dataset type for training. Use `generic` as type if you want to use your own dataset.
    * Choices: ["unlearncanvas", "i2p", "generic"]
    * Example: "unlearncanvas"

* **template:** Type of template to use during training.
    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* **template_name:** Name of the concept or style to erase.
    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"

### Output Configurations

* **output_dir:** Directory to save fine-tuned models and results.
    * Type: str
    * Example: "outputs/concept_ablation/finetuned_models"

### Device Configuration

* **devices:** CUDA devices for training (comma-separated).
    * Type: str
    * Example: "0"




This section provides instructions for running the **evaluation framework** for the Concept ablation algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.

#### **Running the Evaluation Framework**

Create a file, eg, `evaluate.py` and use examples and modify your configs to run the file. Work within the same environment used to perform unlearning for evaluation as well.


**Before running evaluation, download the classifier ckpt from here:**

https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61 

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
                          seed_list = ["188"] )
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


#### **Description of parameters in evaluation_config**

The `evaluation_config`  contains the necessary parameters for running the Concept ablation evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/concept_ablation/finetuned_models/concept_ablation_Abstractionism_model.pth"`

- classification_model : Specifies the classification model used for evaluating the generated outputs.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`

---

### **Training and Sampling Parameters:**

- devices : CUDA device IDs to be used for the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"0"`  

- cfg_text : Classifier-free guidance scale value for image generation. Higher values increase the strength of the conditioning prompt.  
   - *Type:* `float`  
   - *Example:* `9.0`  

- seed : Random seed for reproducibility of results.  
   - *Type:* `int`  
   - *Example:* `188`

- ddim_steps : Number of steps for the DDIM (Denoising Diffusion Implicit Models) sampling process.  
   - *Type:* `int`  
   - *Example:* `100`

- ddim_eta : DDIM eta value for controlling the amount of randomness during sampling. Set to `0` for deterministic sampling.  
   - *Type:* `float`  
   - *Example:* `0.0`

- image_height : Height of the generated images in pixels.  
   - *Type:* `int`  
   - *Example:* `512`

- image_width : Width of the generated images in pixels.  
   - *Type:* `int`  
   - *Example:* `512`

---

### **Output and Logging Parameters:**
- sampler_output_dir : Directory where generated images will be saved during evaluation.  
   - *Type:* `str`  
   - *Example:* `"outputs/eval_results/mu_results/concept_ablation/"`

- use_sample: If you want to just run on sample dataset then set it as True. By default it is True.
   - *Type:* `bool`  
   - *Example:* `True`

---

