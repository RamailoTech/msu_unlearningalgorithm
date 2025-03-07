# Forget Me Not Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The Forget Me Not algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

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
3. Download best.onnx file

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

## Run Train
Create a file, eg, `my_trainer.py` and use examples and modify your configs to run the file.  

1. **Train a Text Inversion using quick canvas dataset**

Before finetuning the model you need to generate safetensors.

```python

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_ti_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_ti_mu,
    ckpt_path="models/diffuser/style50",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ), 
    steps=10,
    template_name = "Abstractionism", #concept to erase
    dataset_type = "unlearncanvas" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/forget_me_not/finetuned_models" #output dir to save finetuned models
)
algorithm.run(train_type="train_ti")
```

**Running the Script in Offlikne Mode**

```bash
WANDB_MODE=offline python my_trainer_ti.py
```

2. **Perform Unlearning using quick canvas dataset**

Before running the `train_attn` script, update the `ti_weights_path` parameter in the configuration file to point to the output generated from the Text Inversion (train_ti.py) stage

```python
from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_attn_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_attn_mu,
    ckpt_path="models/diffuser/style50",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    steps=10,
    ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors",
    template_name = "Abstractionism", #concept to erase
    dataset_type = "unlearncanvas" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/forget_me_not/finetuned_models" #output dir to save finetuned models
)
algorithm.run(train_type="train_attn")
```

1. **Train a Text Inversion using i2p dataset**

Before finetuning the model you need to generate safetensors.

```python

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_ti_i2p,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_ti_i2p,
    ckpt_path="models/diffuser/style50",
    raw_dataset_dir = "data/i2p-dataset/sample",
    steps=10,
    template_name = "self-harm", #concept to erase
    dataset_type = "i2p" ,
    use_sample = True, #train on sample dataset
    output_dir = "outputs/forget_me_not/finetuned_models" #output dir to save finetuned models
)
algorithm.run(train_type="train_ti")
```

**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python my_trainer_ti.py
```

2. **Perform Unlearning**

Before running the `train_attn` script, update the `ti_weights_path` parameter in the configuration file to point to the output generated from the Text Inversion (train_ti.py) stage

```python
from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_train_attn_mu,
)

algorithm = ForgetMeNotAlgorithm(
    forget_me_not_train_attn_mu,
    ckpt_path="models/diffuser/style50",
    raw_dataset_dir=(
        "data/quick-canvas-dataset/sample"
    ),
    steps=10,
    ti_weights_path="outputs/forget_me_not/finetuned_models/Abstractionism/step_inv_10.safetensors",
    use_sample = True, #train on sample dataset
    output_dir = "outputs/forget_me_not/finetuned_models" ,#output dir to save finetuned models
    template_name = "self-harm", #concept to erase
    dataset_type = "i2p" ,
)
algorithm.run(train_type="train_attn")
```


**Running the Script in Offlikne Mode**

```bash
WANDB_MODE=offline python my_trainer_attn.py
```

**How It Works** 
* Default Values: The script first loads default values from the train config file as in configs section.

* Parameter Overrides: Any parameters passed directly to the algorithm, overrides these configs.

* Final Configuration: The script merges the configs and convert them into dictionary to proceed with the training. 

## Directory Structure

- `algorithm.py`: Implementation of the Forget Me NotAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Forget Me NotModel class.
- `scripts/train.py`: Script to train the Forget Me Not algorithm.
- `trainer.py`: Implementation of the Forget Me NotTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---

**This method involves two stages:**

1. **Train a Text Inversion**: The first stage involves training a Text Inversion. Refer to the script [`train_ti.py`](mu/algorithms/forget_me_not/scripts/train_ti.py) for details and implementation. It uses `train_ti_config.yaml` as config file.

2. **Perform Unlearning**: The second stage uses the outputs from the first stage to perform unlearning. Refer to the script [`train_attn.py`](mu/algorithms/forget_me_not/scripts/train_attn.py) for details and implementation. It uses `train_attn_config.yaml` as config file.



### Description of Arguments in train_ti_config.yaml

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Training Configuration**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **steps**: Number of training steps.
- **lr**: Learning rate for the training optimizer.
- **weight_decay_ti**: Weight decay for Text Inversion training.
- **seed**: Random seed for reproducibility.
- **placeholder_tokens**: Tokens used as placeholders during training.
- **placeholder_token_at_data**: Placeholders used in the dataset for Text Inversion training.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_batch_size**: Batch size for training.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).



### Description of Arguments in train_attn_config.yaml

### Key Parameters

**Pretrained Model**

- **ckpt_path**: File path to the pretrained model's checkpoint file.

**Dataset**

- **raw_dataset_dir**: Directory containing the original dataset organized by themes and classes.
- **processed_dataset_dir**: Directory where the processed datasets will be saved.
- **dataset_type**: Type of dataset to use (e.g., `unlearncanvas`).
- **template**: Type of template to use (e.g., `style`).
- **template_name**: Name of the template, defining the style or theme (e.g., `Abstractionism`).
- **use_sample**: Boolean indicating whether to use the sample dataset for training.

**Text Inversion**

- **use_ti**: Boolean indicating whether to use Text Inversion weights.
- **ti_weights_path**: File path to the Text Inversion model weights.

**Tokens**

- **initializer_tokens**: Tokens used to initialize the training process, referencing the template name.
- **placeholder_tokens**: Tokens used as placeholders during training.

**Training Configuration**

- **mixed_precision**: Precision type to use during training (e.g., `fp16` or `fp32`).
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating weights.
- **train_text_encoder**: Boolean to enable or disable training of the text encoder.
- **enable_xformers_memory_efficient_attention**: Boolean to enable memory-efficient attention mechanisms.
- **gradient_checkpointing**: Boolean to enable or disable gradient checkpointing.
- **allow_tf32**: Boolean to allow TensorFloat-32 computation for faster training.
- **scale_lr**: Boolean indicating whether to scale the learning rate based on batch size.
- **train_batch_size**: Batch size for training.
- **use_8bit_adam**: Boolean to enable or disable 8-bit Adam optimizer.
- **adam_beta1**: Beta1 parameter for the Adam optimizer.
- **adam_beta2**: Beta2 parameter for the Adam optimizer.
- **adam_weight_decay**: Weight decay for the Adam optimizer.
- **adam_epsilon**: Epsilon value for the Adam optimizer.
- **size**: Image resolution size for training.
- **with_prior_preservation**: Boolean indicating whether to use prior preservation during training.
- **num_train_epochs**: Number of training epochs.
- **lr_warmup_steps**: Number of steps for linear warmup of the learning rate.
- **lr_num_cycles**: Number of cycles for learning rate scheduling.
- **lr_power**: Exponent to control the shape of the learning rate curve.
- **max-steps**: Maximum number of training steps.
- **no_real_image**: Boolean to skip using real images in training.
- **max_grad_norm**: Maximum norm for gradient clipping.
- **checkpointing_steps**: Number of steps between model checkpoints.
- **set_grads_to_none**: Boolean to set gradients to None instead of zeroing them out.
- **lr**: Learning rate for the training optimizer.

**Output Configuration**

- **output_dir**: Directory path to save training results, including models and logs.

**Device Configuration**

- **devices**: CUDA devices to train on (comma-separated).

**Miscellaneous**

- **only-xa**: Boolean to enable additional configurations specific to the XA pipeline.


#### forget_me_not Evaluation Framework

This section provides instructions for running the **evaluation framework** for the forget_me_not algorithm on Stable Diffusion models. The evaluation framework is used to assess the performance of models after applying machine unlearning.


#### **Running the Evaluation Framework**

You can run the evaluation framework using the `evaluate.py` script located in the `mu/algorithms/forget_me_not/scripts/` directory. Work within the same environment used to perform unlearning for evaluation as well.


### **Basic Command to Run Evaluation:**

**Before running evaluation, download the classifier ckpt from here:**

https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61 

Add the following code to `evaluate.py`

```python
from mu.algorithms.forget_me_not import ForgetMeNotEvaluator
from mu.algorithms.forget_me_not.configs import (
    forget_me_not_evaluation_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.fid import fid_score

evaluator = ForgetMeNotEvaluator(
    forget_me_not_evaluation_config,
    ckpt_path="outputs/forget_me_not/finetuned_models/Abstractionism",
)
generated_images_path = evaluator.generate_images()

reference_image_dir = "/home/ubuntu/Projects/Palistha/testing/data/quick-canvas-dataset/sample"

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


**Run the script**

```bash
python evaluate.py
```


#### **Description of parameters in evaluation_config.yaml**

The `evaluation_config.yaml` file contains the necessary parameters for running the forget_me_not evaluation framework. Below is a detailed description of each parameter along with examples.

---

### **Model Configuration:**
- ckpt_path : Path to the finetuned Stable Diffusion checkpoint file to be evaluated.  
   - *Type:* `str`  
   - *Example:* `"outputs/forget_me_not/finetuned_models/forget_me_not_Abstractionism_model.pth"`

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

- cfg_text_list : Classifier-free guidance scale value for image generation. Higher values increase the strength of the conditioning prompt.  
   - *Type:* `list`  
   - *Example:* `[9.0]`  

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
   - *Example:* `"outputs/eval_results/mu_results/forget_me_not/"`

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







