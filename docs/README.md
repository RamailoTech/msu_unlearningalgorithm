# Unlearn Diff

Unlearn Diff is an open-source Python package designed to streamline the development of unlearning algorithms and establish a standardized evaluation pipeline for diffusion models. It provides researchers and practitioners with tools to implement, evaluate, and extend unlearning algorithms effectively.

### [**Documentation**](https://ramailotech.github.io/msu_unlearningalgorithm/)
You can find the full documentation for this project at the url given below.
https://ramailotech.github.io/msu_unlearningalgorithm/
## Features

- **Comprehensive Algorithm Support**: Includes commonly used concept erasing and machine unlearning algorithms tailored for diffusion models. Each algorithm is encapsulated and standardized in terms of input-output formats.

- **Automated Evaluation**: Supports automatic evaluation on datasets like UnlearnCanvas or IP2P. Performs standard and adversarial evaluations, outputting metrics as detailed in UnlearnCanvas and UnlearnDiffAtk.

- **Extensibility**: Designed for easy integration of new unlearning algorithms, attack methods, defense mechanisms, and datasets with minimal modifications.


### Supported Algorithms

The initial version includes established methods benchmarked in UnlearnCanvas and defensive unlearning techniques:

- **CA** (Concept Ablation)
- **ED** (Erase Diff)
- **ESD** (Efficient Substitution Distillation)
- **FMN** (Forget Me Not)
- **SU** (Saliency Unlearning)
- **SH** (ScissorHands)
- **SA** (Selective Amnesia)
- **SPM** (Semi Permeable Membrane)
- **UCE** (Unified Concept Editing)
For detailed information on each algorithm, please refer to the respective `README.md` files located inside `mu/algorithms`.

## Project Architecture

The project is organized to facilitate scalability and maintainability.

```
.
├── data/
│   ├── i2p-dataset/
│   │   ├── sample/
│   │   └── full/
│   └── unlearn-canvas-dataset/ # Renamed for clarity
│       ├── sample/
│       └── full/
│
├── docs/                 # Documentation, API references, user guides
│
├── models/               # Pre-trained model weights (e.g., Stable Diffusion, LORAs)
│
├── notebooks/            # Example notebooks and experimental code (merges 'examples/')
│
├── outputs/              # All generated outputs (images, artifacts) from runs
│
├── scripts/              # Standalone utility scripts
│   ├── download_models.py
│   └── prepare_datasets.py
│
├── src/                  # Centralized source code for the entire application
│   │
│   ├── mu/         # Core unlearning logic (previously 'mu/')
│   │   ├── __init__.py
│   │   ├── algorithms/
│   │   │   ├── esd/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── algorithm.py      # Core ESD implementation
│   │   │   │   ├── configs/          # ESD-specific configs
│   │   │   │   ├── model.py          # ESD-specific model
│   │   │   │   ├── trainer.py        # ESD-specific trainer
│   │   │   │   └── utils.py          # ESD-specific utilities
│   │   │   └── ca/
│   │   │       └── ... # etc.
│   │   ├── core/               # Base classes shared across unlearning algorithms
│   │   │   ├── base_algorithm.py
│   │   │   ├── base_trainer.py
│   │   │   └── base_model.py
│   │   └── datasets/               # Data handling and loading modules
│   │       ├── __init__.py
│   │       ├── base_handler.py
│   │       └── unlearn_canvas.py
│   │
│   ├── attack/             # Attack logic (previously 'mu_attack/')
│   │   ├── __init__.py
│   │   ├── algorithms/         # Specific attack implementations (previously 'attackers/')
│   │   ├── configs/
│   │   │   ├── illegal/
│   │   │   └── ...
│   │   └── tasks/              # Attack tasks (nudity, violence, etc.)
│   │
│   ├── defense/            # Defense logic (previously 'mu_defense/')
│   │   ├── __init__.py
│   │   └── algorithms/
│   │       └── adv_unlearn/
│   │           ├── __init__.py
│   │           ├── algorithm.py
│   │           ├── configs/
│   │           ├── model.py
│   │           ├── trainer.py
│   │           └── image_generator.py
│   │
│   ├── evaluation/         # Centralized evaluation framework
│   │   ├── __init__.py
│   │   ├── evaluator.py      # Main evaluator runner script
│   │   ├── metrics/
│   │   │   ├── accuracy.py
│   │   │   ├── asr.py
│   │   │   ├── clip.py
│   │   │   └── fid.py
│   │   └── utils/              # Evaluation-specific utilities (e.g., log parsers)
│   │       └── parser.py
│   │
│   └── utils/                # Project-wide shared utilities
│       ├── __init__.py
│       ├── logger.py
│       └── path_setup.py
│
├── tests/                # All tests for the source code
│   ├── test_unlearning.py
│   ├── test_attack.py
│   └── test_evaluation.py
│
├── environment.yaml      # Project-wide environment dependencies
├── main.py               # Main entry point to run training, evaluation, etc.
└── README.md             # Project overview and setup instructions
```


## Datasets

We use the Unlearn Canvas benchmark dataset, available [here](https://huggingface.co/datasets/nebulaanish/quick-canvas-benchmark). Currently, the algorithms are trained using 5 images belonging to the themes of **Abstractionism** and **Architectures**.


## Usage
This section contains the usage guide for the package.

### Installation 

#### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

Before installing the unlearn_diff package, follow these steps to set up your environment correctly. These instructions ensure compatibility with the required dependencies, including Python, PyTorch, and ONNX Runtime.


**Step-by-Step Setup:**

1. Create a Conda Environment Create a new Conda environment named myenv with Python 3.8.5:

```bash
conda create -n myenv python=3.8.5
```

2. Activate the Environment Activate the environment to work within it:

```bash
conda activate myenv
```

3. Install Core Dependencies Install PyTorch, torchvision, CUDA Toolkit, and ONNX Runtime with specific versions:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 onnxruntime==1.16.3 -c pytorch -c conda-forge
```

4. Install our unlearn_diff Package using pip:

```bash
pip install unlearn_diff
```

5. Install Additional Git Dependencies:

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

### Run Train <br>

Each algorithm has their own script to run the algorithm, Some also have different process all together. Follow usage section in readme for the algorithm you want to run with the help of the github repository. You will need to run the code snippet provided in usage section with necessary configuration passed. 


**Example usage for erase_diff algorithm (CompVis model)**

The default configuration for training is provided by erase_diff_train_mu. You can run the training with the default settings as follows:

**Using the Default Configuration**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

algorithm = EraseDiffAlgorithm(
    erase_diff_train_mu
)
algorithm.run()
```

<br> <br>

**Overriding the Default Configuration**

If you need to override the existing configuration settings, you can specify your custom parameters (such as ckpt_path and raw_dataset_dir) directly when initializing the algorithm. For example:

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_mu

algorithm = EraseDiffAlgorithm(
    erase_diff_train_mu,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt", #replace it with your ckpt path
    raw_dataset_dir="data/quick-canvas-dataset/sample",
    use_sample = True, #uses sample dataset
    template_name = "Abstractionism",
    dataset_type = "unlearncanvas",
    devices = "0"
)
algorithm.run()
```

<span style="color: red;"><br>Note: When fine-tuning the model, if you want to use a sample dataset, set use_sample=True (default).Otherwise, set use_sample=False to use the full dataset.<br></span>


**Machine unlearning with i2p dataset**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_i2p

algorithm = EraseDiffAlgorithm(
    erase_diff_train_i2p,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt", #replace it with your ckpt path
    raw_dataset_dir="data/i2p-dataset/sample",
    num_samples = 1,
    dataset_type = "i2p",
    template = "i2p",
    template_name = "self-harm",
    use_sample = True, #uses sample dataset
    devices = "0"
    
)
algorithm.run()
```

**Run on your own dataset**

**Step-1: Generate your own dataset**

```bash
generate_images_for_prompts --model_path models/diffuser/style50 --csv_path data/prompts/generic_data.csv
```

Note:

* generate_images_for_prompts: This command invokes the image generation script. It uses a diffusion model to generate images based on textual prompts.

* --model_path: Specifies the path to the diffusion model to be used for image generation. In this example, the model is located at models/diffuser/style50.

* --csv_path: Provides the path to a CSV file containing the prompts. Each prompt in this CSV will be used to generate an image, allowing you to build a dataset tailored to your needs.


**Step-2: Train on your own dataset**

```python
from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs import erase_diff_train_i2p

algorithm = EraseDiffAlgorithm(
    erase_diff_train_i2p,
    ckpt_path="/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt", #replace it with your ckpt path
    raw_dataset_dir="data/generic_data",
    num_samples = 1,
    dataset_type = "generic", #add the dataset type as generic
    template_name = "self-harm", #concept to erase
    use_sample = True, #uses sample dataset
    devices = "0"
    
)
algorithm.run()
```



### Evaluation:

1. **Evaluate using unlearn canvas dataset:**

Note: Currently it supports evaluation for unlearn canvas dataset. I2p and generic dataset support needs to be added.

```python
from mu.algorithms.erase_diff import EraseDiffEvaluator
from mu.algorithms.erase_diff.configs import (
    erase_diff_evaluation_config
)
from evaluation.metrics.accuracy import accuracy_score
from evaluation.metrics.clip import clip_score
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

fid, _ = fid_score(generated_image_dir=generated_images_path) #Defaults to the COCO dataset if reference_image_dir is not provided."
print(fid)

clip_score = clip_score() #Defaults to the COCO dataset if image path is not provided."
print(clip_score)
```



**Link to our example usage notebooks**

1. **Erase-diff (compvis model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_erase_diff.ipynb

2. **forget-me-not (Diffuser model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_forget_me_not.ipynb
