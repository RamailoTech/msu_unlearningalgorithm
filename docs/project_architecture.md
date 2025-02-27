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