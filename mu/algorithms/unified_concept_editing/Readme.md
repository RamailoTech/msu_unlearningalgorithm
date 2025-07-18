# Unified Concept Editing Algorithm for Machine Unlearning

This repository provides an implementation of the unified concept editing algorithm for machine unlearning in Stable Diffusion models. The unified concept editing algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

```bash
conda --version
```

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
pip install git+https://github.com/CompVis/taming-transformers.git@master git+https://github.com/openai/CLIP.git@main git+https://github.com/crowsonkb/k-diffusion.git git+https://github.com/cocodataset/panopticapi.git git+https://github.com/Phoveran/fastargs.git@main git+https://github.com/boomb0om/text2image-benchmark
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
  - **unlearn_canvas**:
    - **Sample**:
     ```
     download_data sample unlearn_canvas
     ```
    - **Full**:
     ```
     download_data full unlearn_canvas
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

4. **Download coco dataset**

  ```
  download_coco_dataset
  ```

**Verify the Downloaded Files**

After downloading, verify that the datasets have been correctly extracted:
```bash
ls -lh ./data/i2p-dataset/sample/
ls -lh ./data/quick-canvas-dataset/sample/
```
---

## Usage

To train the Unified Concept Editing algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

**Using Unlearn Canvas dataset**

```python
from mu.algorithms.unified_concept_editing.algorithm import (
    UnifiedConceptEditingAlgorithm,
)
from mu.algorithms.unified_concept_editing.configs import (
    unified_concept_editing_train_mu,
)

algorithm = UnifiedConceptEditingAlgorithm(
    unified_concept_editing_train_mu,
    ckpt_path="models/diffuser/style50/",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
)
algorithm.run()
```

**Using i2p dataset**

```python
from mu.algorithms.unified_concept_editing.algorithm import (
    UnifiedConceptEditingAlgorithm,
)
from mu.algorithms.unified_concept_editing.configs import (
    unified_concept_editing_train_i2p,
)

algorithm = UnifiedConceptEditingAlgorithm(
    unified_concept_editing_train_i2p,
    ckpt_path="models/diffuser/style50/",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    output_dir="/opt/dlami/nvme/outputs",
    use_sample = True # to run on sample dataset
    dataset_type = "i2p",
    template_name = "self-harm",
)
algorithm.run()
```


**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python my_trainer.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

### Directory Structure

- `algorithm.py`: Implementation of the ScissorHandsAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the ScissorHandsModel class.
- `scripts/train.py`: Script to train the ScissorHands algorithm.
- `trainer.py`: Implementation of the ScissorHandsTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class
---


### Description of Arguments in train_config.yaml
**Training Parameters**

* **train_method**: Specifies the method of training for concept erasure.
    * Choices: ["full", "partial"]
    * Example: "full"

* **alpha**: Guidance strength for the starting image during training.
    * Type: float
    * Example: 0.1

* **epochs**: Number of epochs to train the model.
    * Type: int
    * Example: 10

* **lr**: Learning rate used for the optimizer during training.
    * Type: float
    * Example: 5e-5


**Model Configuration**
* **ckpt_path**: File path to the checkpoint of the Stable Diffusion model.
    * Type: str
    * Example: "/path/to/model_checkpoint.ckpt"

* **config_path**: File path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/config.yaml"

**Dataset Directories**

* **dataset_type**: Specifies the dataset type for the training process.
    * Choices: ["unlearncanvas", "i2p"]
    * Example: "unlearncanvas"

* **template**: Type of template to use during training.
    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* **template_name**: Name of the specific concept or style to be erased.
    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"

**Output Configurations**

* **output_dir**: Directory where the fine-tuned models and results will be saved.
    * Type: str
    * Example: "outputs/erase_diff/finetuned_models"

**Sampling and Image Configurations**

* **use_sample**: Flag to indicate whether a sample dataset should be used for training.
    * Type: bool
    * Example: True

* **guided_concepts**: Concepts to guide the editing process.
    * Type: str
    * Example: "Nature, Abstract"

* **technique**: Specifies the editing technique.
    * Choices: ["replace", "tensor"]
    * Example: "replace"

* **preserve_scale**: Scale for preservation during the editing process.
    * Type: float
    * Example: 0.5

* **preserve_number**: Number of items to preserve during editing.
    * Type: int
    * Example: 10

* **erase_scale**: Scale for erasure during the editing process.
    * Type: float
    * Example: 0.8

* **lamb**: Lambda parameter for controlling balance during editing.
    * Type: float
    * Example: 0.01

* **add_prompts**: Flag to indicate whether additional prompts should be used.
    * Type: bool
    * Example: True

**Device Configuration**

* **devices**: Specifies the CUDA devices to be used for training (comma-separated).
    * Type: str (Comma-separated)
    * Example: "0,1"


#### unified_concept_editing Evaluation Framework

This section provides instructions for running the **evaluation framework** for the unified_concept_editing algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/unified_concept_editing/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


**Before running evaluation, download the classifier ckpt from here:**

https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61 


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

**Running in Offline Mode:**

```bash
WANDB_MODE=offline python evaluate.py
```

#### **Description of parameters in evaluation_config.yaml**

The `evaluation_config.yaml` file contains the necessary parameters for running the unified_concept_editing evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/unified_concept_editing/finetuned_models/unified_concept_editing_Abstractionism_model.pth"`

- classification_model : Specifies the classification model used for evaluating the generated outputs.  
   - *Type:* `str`  
   - *Example:* `"vit_large_patch16_224"`
---

### **Training and Sampling Parameters:**
- theme : Specifies the theme or concept being evaluated for removal from the model's outputs.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`

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
   - *Example:* `"outputs/eval_results/mu_results/unified_concept_editing/"`
---

### **Optimization Parameters:**
- forget_theme : Concept or style intended for removal in the evaluation process.  
   - *Type:* `str`  
   - *Example:* `"Bricks"`

- seed_list : List of random seeds for performing multiple evaluations with different randomness levels.  
   - *Type:* `list`  
   - *Example:* `["188"]`

- use_sample: If you want to just run on sample dataset then set it as True. By default it is True.
   - *Type:* `bool`  
   - *Example:* `True`

