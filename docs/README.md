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

- **`mu/`**: Core source code for machine unlearning.
  - **`algorithms/`**: Implementation of various algorithms. Each algorithm has its own subdirectory containing code and a `README.md` with detailed documentation.
    - **`esd/`**: ESD algorithm components.
      - `README.md`: Documentation specific to the ESD algorithm.
      - `algorithm.py`: Core implementation of ESD.
      - `configs/`: Configuration files for training and generation tasks.
      - `constants/const.py`: Constant values used across the ESD algorithm.
      - `environment.yaml`: Environment setup for ESD.
      - `model.py`: Model architectures specific to ESD.
      - `sampler.py`: Sampling methods used during training or inference.
      - `evaluator.py`: Evaluation of ESD algorithm.
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
    - `base_config.py`: Base class for config class.
  - **`datasets/`**: Dataset management and utilities.
    - `__init__.py`: Initializes the dataset package.
    - `dataset.py`: Dataset classes and methods.
    - `helpers/`: Helper functions for data processing.
    - `unlearning_canvas_dataset.py`: Specific dataset class for unlearning tasks.
  - **`helpers/`**: Utility functions and helpers.
    - `helper.py`: General-purpose helper functions.
    - `logger.py`: Logging utilities to standardize logging practices.
    - `path_setup.py`: Path configurations and environment setup.
- **`mu_attack/`**: Core source code for attack.
  - **`attackers/`**: Implementation of each of the attacks.
    - **`hard_prompt.py`**: Implementation for hard prompt attack.
    - **`no_attack.py`**: Implementation for no attack.
    - **`random.py`**: Implementation for random attack.
    - **`seed_search.py`**: Implementation for seed search attack.
    - **`soft_prompt.py`**: Implementation for soft_prompt attack.
    - **`text_grad.py`**: Implementation for text_grad attack.
  - **`configs`**: Configuration files for attackers and evaluation tasks.
    - **`adv_unlearn`**: Config class for adversial unlearning.
    - **`evaluation`**: Config class for evaluation.
    - **`illegal`**: Config class for illegal.
    - **`nudity`**: Config class for nudity.
    - **`object`**: Config class for object.
    - **`style`**: Config class for style.
    - **`violence`**: Config class for violence.
  - **`core/`**: Foundational classes and utilities.
    - **`base_attacker.py`**: Abstract base class for attack implementations.
    - **`base_config.py`**: Base class for config class.
    - **`base_stable_diffusion.py`**: Base class for stable diffusion pipeline.
    - **`base_task.py`**: Base class for tasks.
  - **`datasets/`**: Script to generate dataset.
  - **`exces/`**: Implemenation of attacks.
    - **`adv_attack.py`** : Implementation of advattack.
    - **`attack.py`** : Main orchestration class that implements attacks.
  - **`helpers/`**: Utility functions and helpers.
    - **`utils.py`**: Contains utility functions used in mu_attack.
  - **`loggers/`**: Logging utilities to standardize logging practices.
  - **`tasks/`**: Contains tasks for attacks.
    - **`utils/`**: Utilities for tasks.
    - **`classifier.py`**: Implementation of classifier task.
    - **`p4d.py`**: Implementation of p4d task.
    - **`transfer.py`**: Implementation of transfer task.
    - **`sd_compvis.py`**: Base stable diffusion pipeline implementation for compvis model.
    - **`sd_diffuser.py`**: Base stable diffusion pipeline implementation for diffuser model.
  - `environment.yaml`: Environment setup for attack.
- **`mu_defense/`**: Core source code for defense i.e adversial unlearning.
  - **`algorithms/`**: Algorithms for defense.
    - **`adv_unlearn/`**: Implementation of defense algorithm (adv_unlearn).
      - **`configs/`**: Configuration files for AdvUnlearn, evaluation and image generation tasks.
        - **`adv_unlearn_config.py`**: Configuration class for AdvUnlearn.
        - **`evaluation_config.py`**: Configuration class for evaluation for AdvUnlearn.
        - **`example_img_generator_config.py`**: Configuration class for image generation for AdvUnlearn evaluation.
      - `algorithm.py`: Core implementation of AdvUnlearn.
      - `trainer.py`: Core training routines and optimization strategies.
      - `compvis_trainer.py`: Compvis specific training routines and optimization strategies.
      - `diffuser_trainer.py`: Diffuser specific training routines and optimization strategies.
      - `generate_example_image.py`: Implementation of image generation for evaluation for both compvis and diffuser.
      - `evaluator.py`: Implementation of evaluation framework for adv_unlearn.
      - `model.py`: Script to load Compvis and diffuser model.
      - `utils.py`: Utility function used in AdvUnlearn.
      - `README.md`: Docs for AdvUnlearn.
    - **`core/`**: Foundational classes and utilities.
      - `base_algorithm.py`: Abstract base class for algorithm implementations.
      - `base_data_handler.py`: Base class for data handling.
      - `base_model.py`: Base class for model definitions.
      - `base_config.py`: Base class for config class.
      - `base_trainer.py`: Base class for training routines.
    - `environment.yaml`: Environment setup for defense.
- **`tests/`**: Test suites for ensuring code reliability.
- **`stable_diffusion/`**: Components for stable diffusion.
- **`lora_diffusion/`**: Components for the LoRA Diffusion.


## Datasets

We use the Quick Canvas benchmark dataset, available [here](https://huggingface.co/datasets/nebulaanish/quick-canvas-benchmark). Currently, the algorithms are trained using 5 images belonging to the themes of **Abstractionism** and **Architectures**.




## Usage
This section contains the usage guide for the package.

### Installation
```
pip install unlearn_diff
```
### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

```bash
conda --version
```
### Create environment (Default):
```
create_env
```
#### Optional(Create environment for algorithm):
```
create_env erase_diff
```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate UnlearnDiff```

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

