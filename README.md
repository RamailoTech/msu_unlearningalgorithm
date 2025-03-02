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

- **`data/`**: Stores data-related files.
  - **`i2p-dataset/`**: contains i2p-dataset
    - **`sample/`**: Sample dataset
    - **`full/`**: Full dataset

  - **`quick-canvas-dataset/`**: contains quick canvas dataset
    - **`sample/`**: Sample dataset
    - **`full/`**: Full dataset

- **`docs/`**: Documentation, including API references and user guides.

- **`outputs/`**: Outputs of the trained algorithms.

- **`examples/`**: Sample code and notebooks demonstrating usage.

- **`logs/`**: Log files for debugging and auditing.

- **`models/`**: Repository of trained models and checkpoints.

- **`mu/`**: Core source code.
  - **`algorithms/`**: Implementation of various algorithms. Each algorithm has its own subdirectory containing code and a `README.md` with detailed documentation.
    - **`esd/`**: ESD algorithm components.
      - `README.md`: Documentation specific to the ESD algorithm.
      - `algorithm.py`: Core implementation of ESD.
      - `configs/`: Configuration files for training and generation tasks.
      - `constants/const.py`: Constant values used across the ESD algorithm.
      - `environment.yaml`: Environment setup for ESD.
      - `model.py`: Model architectures specific to ESD.
      - `sampler.py`: Sampling methods used during training or inference.
      - `scripts/train.py`: Training script for ESD.
      - `trainer.py`: Training routines and optimization strategies.
      - `utils.py`: Utility functions and helpers.
    - **`ca/`**: Components for the CA algorithm.
      - `README.md`: Documentation specific to the CA algorithm.
      - *...and so on for other algorithms*
  - **`core/`**: Foundational classes and utilities.
    - `base_algorithm.py`: Abstract base class for algorithm implementations.
    - `base_data_handler.py`: Base class for data handling.
    - `base_model.py`: Base class for model definitions.
    - `base_sampler.py`: Base class for sampling methods.
    - `base_trainer.py`: Base class for training routines.
  - **`datasets/`**: Dataset management and utilities.
    - `__init__.py`: Initializes the dataset package.
    - `dataset.py`: Dataset classes and methods.
    - `helpers/`: Helper functions for data processing.
    - `unlearning_canvas_dataset.py`: Specific dataset class for unlearning tasks.
  - **`helpers/`**: Utility functions and helpers.
    - `helper.py`: General-purpose helper functions.
    - `logger.py`: Logging utilities to standardize logging practices.
    - `path_setup.py`: Path configurations and environment setup.

- **`tests/`**: Test suites for ensuring code reliability.
- **`stable_diffusion/`**: Components for stable diffusion.
- **`lora_diffusion/`**: Components for the LoRA Diffusion.

## Datasets

We use the Quick Canvas benchmark dataset, available [here](https://huggingface.co/datasets/nebulaanish/quick-canvas-benchmark). Currently, the algorithms are trained using 5 images belonging to the themes of **Abstractionism** and **Architectures**.




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


#### Optional(Create environment for algorithm):

If you want to create algorithm specific environment then use command given below:

```bash
create_env erase_diff
```

The <algorithm_name> has to be one of the folders in the `mu/algorithms` folder.



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


### Run Train <br>

Each algorithm has their own script to run the algorithm, Some also have different process all together. Follow usage section in readme for the algorithm you want to run with the help of the github repository. You will need to run the code snippet provided in usage section with necessary configuration passed. 


**Link to our example usage notebooks**

1. **Erase-diff (compvis model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_erase_diff.ipynb

2. **forget-me-not (Diffuser model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_forget_me_not.ipynb
