# Contributing to Unlearn Diff

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

## Introduction

**Unlearn** is an open-source Python package designed for unlearning algorithms in diffusion models. We welcome contributions from the community, whether it’s a new feature, bug fix, or an entirely new unlearning algorithm.

---

## Code of Conduct

Please note that we expect all contributors to abide by our code of conduct. Be respectful and considerate of others, and help keep our community healthy and inclusive.

---

## Project Structure

A quick overview of relevant folders:

- **`mu/algorithms/`**  
  This is where **all algorithms** reside. Each algorithm has its own subfolder containing:
  - `algorithm.py` (core logic)
  - `data_handler.py` (if data-processing is needed)
  - `datasets/` (special dataset classes)
  - `environment.yaml` (conda environment details)
  - `evaluator.py` (evaluation routines)
  - `model.py` (model definitions)
  - `sampler.py` (sampling methods)
  - `scripts/` (training, evaluation scripts, etc.)
  - `trainer.py` (training loop, optimization)
  - `utils.py` (auxiliary functions)
  - `Readme.md` (algorithm-specific documentation)

- **`mu/core/`**  
  Contains abstract/base classes (`base_algorithm.py`, `base_trainer.py`, etc.) that algorithms can extend or override.

- **`mu/datasets/`**  
  Common dataset classes and utilities used by multiple algorithms.

- **`helpers/`**  
  Shared utility functions (e.g., logging, path setup, config loading).

- **`mu_attack/` & `mu_defense/`**:  
  These directories contain the implementation of attack strategies and defensive unlearning mechanisms, respectively. Each contains:
  - **Attack Modules (`mu_attack/`)**:
    - `attackers/`: Different attack implementations (e.g., hard/soft prompts, seed search).
    - `configs/`: YAML configuration files for attack routines.
    - `tasks/`: Task definitions to evaluate attack efficacy.
  - **Defense Modules (`mu_defense/`)**:
    - `algorithms/`: Defensive algorithms including adversarial unlearning methods.
    - `configs/`: Configurations for training and evaluating defenses.

- **Other Directories**:
  - **`data/`**: Contains datasets.
  - **`docs/`**: Project documentation and API references.
  - **`outputs/`**: Generated outputs from algorithms.
  - **`logs/`**: Log files for debugging.
  - **`models/`**: Trained models and checkpoints.
  - **`tests/`**: Test suites to ensure code reliability.
  - **`stable_diffusion/`** & **`lora_diffusion/`**: Diffusion components.

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

## Adding a New Algorithm

One of the core goals of **Unlearn Diff** is to make it easy to add and benchmark new unlearning algorithms. Follow these steps when introducing a new method.

### Folder Structure

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

## Adding a New Attack

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

## Adding a New Defense

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


### Creating an Environment

The default environment file is located in the project root directory (`environment.yaml`). Contributors should update this file as needed to include any new packages required by their contributions. If your module or algorithm requires unique dependencies, you may add a dedicated environment file in its respective directory, but be sure to update and maintain the default environment in the root.

Optionally, to keep dependencies organized, each algorithm has its own `environment.yaml`. Contributors are encouraged to:

- Specify all required packages (e.g., `torch`, `torchvision`, `diffusers`, etc.).  
- Name the environment after the algorithm, e.g., `mu_<algorithm_name>`.  
- Update instructions in your algorithm’s `Readme.md` on how to install and activate this environment.

### Documentation

- Each algorithm folder **must** contain a `Readme.md` that documents:
  - High-level description of the algorithm (goals, references to papers, etc.).
  - How to install any special dependencies (if not covered by `environment.yaml`).
  - How to run training and evaluation (with example commands).
  - Explanation of the config files in `configs/`.
  - Any unique parameters or hyperparameters.

- If you introduce new metrics or logs, please outline them in your algorithm’s README and update any relevant docs in the main `docs/` folder as needed.

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