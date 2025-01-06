# Unlearn

Unlearn is an open-source Python package designed to streamline the development of unlearning algorithms and establish a standardized evaluation pipeline for diffusion models. It provides researchers and practitioners with tools to implement, evaluate, and extend unlearning algorithms effectively.


## Features

- **Comprehensive Algorithm Support**: Includes commonly used concept erasing and machine unlearning algorithms tailored for diffusion models. Each algorithm is encapsulated and standardized in terms of input-output formats.

- **Automated Evaluation**: Supports automatic evaluation on datasets like UnlearnCanvas or IP2P. Performs standard and adversarial evaluations, outputting metrics as detailed in UnlearnCanvas and UnlearnDiffAtk.

- **Extensibility**: Designed for easy integration of new unlearning algorithms, attack methods, defense mechanisms, and datasets with minimal modifications.


### Supported Algorithms

The initial version includes established methods benchmarked in UnlearnCanvas and defensive unlearning techniques:

- **ESD** (Efficient Substitution Distillation)
- **CA**
- **UCE**
- **FMN**
- **SalUn**
- **SEOT**
- **SPM**
- **EDiff**
- **ScissorHands**
- *...and more*

For detailed information on each algorithm, please refer to the respective `README.md` files located inside `mu/algorithms`.

## Project Architecture

The project is organized to facilitate scalability and maintainability.

- **`data/`**: Stores data-related files.
  - **`processed_data/`**: Preprocessed data ready for models.
  - **`raw_data/`**: Original datasets.
  - **`results/`**: Outputs from algorithms.
    - **`esd/`**: Results specific to the ESD algorithm.
    - **`algorithm_2/`**: Results from other algorithms.
  - **`images/`**: Generated or processed images.
  - **`models/`**: Saved model checkpoints.

- **`docs/`**: Documentation, including API references and user guides.

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

## Datasets

We use the Quick Canvas benchmark dataset, available [here](https://huggingface.co/datasets/nebulaanish/quick-canvas-benchmark). Currently, the algorithms are trained using 5 images belonging to the themes of **Abstractionism** and **Architectures**.




## Usage
This section contains the usage guide for the package.
### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running:

```bash
conda --version
```

### Downloading data and models.
After you install the package, you can use following commands to download.
1. **Dataset**
<br>
`<dataset_type> : sample | full ` <br>
`<dataset_source>: i2p | quick_canvas`
  ```
  download_data <dataset_type> <dataset_source>
  ```
  eg:  `downlaod_data sample i2p`

2. **Model** <br>
`<model_type> : compvis | diffuser`
  ```
  download_model <model_type>
  ```
  eg: `download_model compvis`