<a id="contribution-section"></a>

**Contributing to Unlearn Diff**

Thank you for your interest in contributing to **Unlearn**! This document outlines the steps and best practices for adding new algorithms, making pull requests, filing issues, and more.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Code of Conduct](#code-of-conduct)  
3. [Project Structure](#project-structure)  
4. [How to Contribute](#how-to-contribute)  
   - [Reporting Issues](#reporting-issues)  
   - [Suggesting Enhancements](#suggesting-enhancements)  
   - [Submitting Pull Requests](#submitting-pull-requests)  
5. [Adding a New Algorithm](#adding-a-new-algorithm)  
   - [Folder Structure](#folder-structure)  
   - [Creating an Environment](#creating-an-environment)  
   - [Documentation](#documentation)  
6. [Code Style](#code-style)  
7. [Contact](#contact)

---

**Introduction**

**Unlearn** is an open-source Python package designed for unlearning algorithms in diffusion models. We welcome contributions from the community, whether it’s a new feature, bug fix, or an entirely new unlearning algorithm.

---

## Code of Conduct

Please note that we expect all contributors to abide by our code of conduct. Be respectful and considerate of others, and help keep our community healthy and inclusive.

---


## Project Architecture

The project is organized to facilitate scalability and maintainability.

- **`data/`**: Stores data-related files.

    - **`quick-canvas-dataset/`**: contains quick canvas dataset
        - **`sample/`**: Sample dataset
        - **`full/`**: Full dataset

- **`docs/`**: Documentation, including API references and user guides.

- **`logs/`**: Log files for debugging and auditing.

- **`models/`**: Repository of lora_diffusion and stable_diffusion.

- **`evaluation/`**: Contains metrics for evalaution.
    - **`core/`**:Foundational classes.
      - **`base_evaluator.py`**: Base class for evaluation.
      - **`mu_defense_base_image_generator.py`**: Base class for image generation.
    - **`helpers/`**: Utility functions and helpers.
      - **`parser.py`**: Parse attack logs for evaluation.
      - **`utils.py`**: Utility function.
    - **`metrics/`**: Contains metrics for evalaution.
      - **`accuracy.py`**
      - **`asr.py`**
      - **`clip.py`**
      - **`fid.py`**

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
            - `evaluator.py`: Script that generates necessary outputs for evaluation.
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
- **`mu_attack/`**: Implementation of attack algorithms.
    - **`attackers/`**: Contains different types of attackers.
    - **`configs/`**: Configurations file.
        - **`illegal/`**: config for illegal task.
        - **`nudity/`**: config for nudity task.
        - **`object/`**: config for object task.
        - **`style/`**: config for style task.
        - **`violence/`**: config for violence task.
    - **`core/`**: Foundational classes.
    - **`datasets/`**: script to generate dataset.
    - **`exces/`**: Script to run attack
    - **`tasks/`**: Implementation of tasks
    - **`helpers/`**: Utility functions

- **`mu_defense/`**: Implementation of Advunlearn algorithms.
  - **`algorithms/`**: Implementation of various defense algorithms. Each algorithm has its own subdirectory containing code and a `README.md` with detailed documentation.
      - **`adv_unlearn/`**: Adversial Unlearn algorithm components.
          - `README.md`: Documentation specific to the advunlearn algorithm.
          - `algorithm.py`: Core implementation of advunlearn.
          - `configs/`: Configuration files for training and generation tasks.
          - `model.py`: Model architectures specific to advunlearn.
          - `image_generator.py`: Image generator methods for generating sample images for evaluation.
          - `evaluator.py`: Script that generates necessary outputs for evaluation.
          - `dataset_handler.py`: Dataset handler for advunlearn algorithm. 
          - `compvis_trainer.py`: training loop for CompVis models.
          - `diffuser_trainer.py`: training loop for diffuser models.
          - `trainer.py`: Trainer class orchestrates the adversarial unlearning training process.
          - `utils.py`: Utility functions and helpers.
- **`scripts/`**: Commands to generate datasets and download models.
- **`notebooks/`**: Contains example implementation.
- **`tests/`**: Contains pytests.

---

## How to Contribute

### Reporting Issues

1. Check the to see if your bug or feature request has already been reported.  
2. If not, open a new issue. Provide a clear title and description, including steps to reproduce if it’s a bug.  
3. Label your issue appropriately (e.g., “bug”, “enhancement”, “documentation”).

### Suggesting Enhancements

- If you have an idea for a new feature or improvement, open a GitHub issue labeled “enhancement” or “feature request.”  
- Include any relevant use-cases, examples, or background information to help the community understand your proposal.

### Submitting Pull Requests

1. **Fork** the repository and create your feature branch from `main` (or the appropriate branch if instructed otherwise):  
   ```bash
   git checkout -b feature/my-awesome-feature
   ```
2. Make your changes, including tests if applicable.  
3. Commit and push your changes:  
   ```bash
   git push origin feature/my-awesome-feature
   ```
4. Open a **Pull Request** from your fork into the main repo. Provide a clear description of what your PR does, referencing any related issues.

Please ensure your PR follows the [Code Style](#code-style) guidelines and includes updated or new tests if needed.

---

#### Adding a New Algorithm

One of the core goals of **Unlearn Diff** is to make it easy to add and benchmark new unlearning algorithms. Follow these steps when introducing a new method.

#### Folder Structure

1. **Create a new subfolder** in `mu/algorithms/` with a clear, descriptive name (e.g., `my_new_algo`).  
2. Inside this subfolder, include the following (at minimum):
    - `algorithm.py`  
    - `trainer.py`  
    - `scripts/train.py` (and optionally `scripts/evaluate.py` if your training and evaluation are separate)  
    - `configs/` containing YAML config files for training, evaluation, etc.  
    - `environment.yaml` specifying dependencies for `conda`  
    - `Readme.md` explaining how to install, run, and configure your new algorithm  
    - (Optional) `data_handler.py`, `datasets/`, `model.py`, `sampler.py`, `utils.py`, etc., depending on your algorithm’s design.

3. **Extend or import** from base classes in `mu/core/` if your algorithm logic aligns with any existing abstract classes (e.g., `BaseAlgorithm`, `BaseTrainer`, etc.). This ensures code consistency and easier maintenance.

#### Adding a New Attack

To add a new attack method, follow these guidelines:

1. **Folder Structure**:
    - Create a new file or subfolder under `mu_attack/attackers/` with a clear, descriptive name (e.g., `my_new_attack.py` or `mu_attack/attackers/my_new_attack/`).
    - If creating a subfolder, include essential files specific to attack implementation:
        - **Attacks**: The main implementation file for the attack logic (e.g., `attack.py`).
        - **Execs**: Scripts or modules that execute the attack routines.
        - **Tasks**: Task definitions for integrating and testing the attack.
        - Any helper modules or configuration files specific to your attack.
    - Update or add corresponding YAML configuration files and config class under `mu_attack/configs/` if your attack requires custom settings.

2. **Implementation**:
    - Extend or import from the base class `BaseAttacker` (located in `mu_attack/core/base_attacker.py`) if applicable.
    - Ensure that your attack method adheres to the input-output standards defined by the project.

3. **Documentation & Testing**:
    - Add detailed documentation within your new attack module and update the main documentation if needed.
    - Include tests covering your new attack method under the appropriate test directories.

4. **Environment**:
    - If your attack method has unique dependencies, update the `environment.yaml` file within the relevant directory or provide instructions in your documentation on how to create a dedicated environment. Also update the common environment file that is located in the project's root directory.

#### Adding a New Defense

To integrate a new defense mechanism, please follow these steps:

1. **Folder Structure**:
    - Create a new subfolder under `mu_defense/algorithms/` with a descriptive name (e.g., `my_new_defense`).
    - Within this subfolder, include essential files such as:
      - `algorithm.py`: Contains the core logic of your defense method.
      - `trainer.py`: Contains training routines and optimization strategies.
      - `configs/`: Include configuration class for training and evaluation.
      - `environment.yaml`: Specify dependencies unique to your defense method.
      - `Readme.md`: Document usage instructions, configuration details, and any other relevant information.

2. **Implementation**:
    - Extend or use base classes provided in `mu_defense/algorithms/core/` (e.g., `base_algorithm.py`, `base_trainer.py`) to ensure consistency with existing methods.
    - Implement any unique evaluation metrics or procedures if your defense requires them.

3. **Documentation & Testing**:
    - Document your defense method thoroughly within its `Readme.md` and update the global documentation if necessary.
    - Provide tests for your defense implementation to ensure its reliability and compatibility with the rest of the system.

4. **Environment**:
    - If your defense algorithm has specific dependencies, use the provided `environment.yaml` file as a template and adjust it accordingly. Include clear instructions for users to create and activate the environment. Also update the common environment file that is located in the project's root directory.


#### Creating an Environment

The default environment file is located in the project root directory (`requirements.txt`). Contributors should update this file as needed to include any new packages required by their contributions making sure it doesnot effect other algorthms. If your module or algorithm requires unique dependencies, you may add a dedicated environment file in its respective directory, but be sure to update and maintain the default environment in the root.

Optionally, to keep dependencies organized, each algorithm has its own `environment.yaml`. Contributors are encouraged to:

- Specify all required packages (e.g., `torch`, `torchvision`, `diffusers`, etc.). 
- Name the environment after the algorithm, e.g., `mu_<algorithm_name>`.  
- Update instructions in your algorithm’s `Readme.md` on how to install and activate this environment.
- Ensure you run the pytests in the tests/ directory after adding a new algorithm to verify that the updated environment remains compatible with existing algorithms.

### Run pytests

Before integrating new algorithms into the repository, please run the existing tests using the modified environment file to ensure that your changes do not break any functionality.

**Steps to Run the Tests**

1. Prepare the Environment Configuration:
    Make sure the environment is set up using the provided tests/test_config.yaml file. This file contains configuration settings for various components of the project, such as data directories, model paths, and algorithm-specific parameters. It includes:

    * common_config_unlearn_canvas_mu:
        Contains settings for the UnlearnCanvas dataset, including paths for the data directory, CompVis and Diffuser model checkpoints, template information, and dataset type.

    * common_config_i2p:
        Provides configuration for the i2p dataset with similar settings as above.

    * evaluator_config:
        Specifies the classifier checkpoint path and sample usage flag for evaluation.

    * Algorithm-Specific Configurations:
        Settings for various algorithms like concept_ablation, forget_me_not, saliency_unlearning, semipermeable, unified_concept_editing, selective_amnesia, and attack/defense configurations.

2. Run the Tests:
    Execute the following commands from the root of your repository:

```bash
pytest tests/test_mu.py
pytest tests/test_mu_attack.py
pytest tests/test_mu_defense.py
```
These commands will run tests for:

* tests/test_mu.py: Core machine unlearning functionalities including evaluation.

* tests/test_mu_attack.py: Attack-related functionalities including evaluation.

* tests/test_mu_defense.py: Defense-related functionalities including evaluation.

3. Verify the Results:
Ensure that all tests pass without errors. If any test fails, fix the issues before adding new algorithms to maintain compatibility with the existing codebase.

### Documentation

- Each algorithm folder **must** contain a `Readme.md` that documents:
  - High-level description of the algorithm (goals, references to papers, etc.).
  - How to install any special dependencies (if not covered by `environment.yaml`).
  - How to run training and evaluation (with example commands).
  - Explanation of the config files in `configs/`.
  - Any unique parameters or hyperparameters.

- If you introduce new metrics or logs, please outline them in your algorithm’s README and update any relevant docs in the main `docs/` folder as needed.

- If you have incorporated code or drawn inspiration from external codebases, please ensure that you properly credit and cite the original work within each script.

---

## Code Style

- We recommend following [PEP 8](https://peps.python.org/pep-0008/) for Python code.  
- Use descriptive variable names, docstrings, and type hints where possible.  
- Keep functions and methods short and focused—avoid large, monolithic methods.  
- Use Python’s built-in logging or the logging utilities in `helpers/logger.py` for consistent log outputs. Use wandb global wandb logger if required.  

Additionally, we prefer:
- **4 spaces** for indentation (no tabs).
- Meaningful commit messages: “Fix dataset loader bug for large images” vs. “Fix stuff.”

---

## Contact

If you have any questions, feel free to open an issue or reach out to us via email. We appreciate your contributions and look forward to collaborating with you on **Unlearn Diff**!

---

Thank you again for your interest in contributing. We’re excited to see your ideas and improvements!  