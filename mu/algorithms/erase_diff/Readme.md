# EraseDiff Algorithm for Machine Unlearning

This repository provides an implementation of the erase diff algorithm for machine unlearning in Stable Diffusion models. The erasediff algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

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
### Create environment:
```
create_env <algorithm_name>
```
eg: ```create_env erase_diff```

### Activate environment:
```
conda activate <environment_name>
```
eg: ```conda activate erase_diff```

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

**Verify the Downloaded Files**

After downloading, verify that the datasets have been correctly extracted:
```bash
ls -lh ./data/i2p-dataset/sample/
ls -lh ./data/quick-canvas-dataset/sample/
```


## Run Train

**Example Command**

```bash
python -m mu.algorithms.erase_diff.scripts.train \
--config_path mu/algorithms/erase_diff/configs/train_config.yaml
```

**Running the Training Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.erase_diff.scripts.train \
--config_path mu/algorithms/erase_diff/configs/train_config.yaml
```

**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.


**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.erase_diff.scripts.train \
--config_path mu/algorithms/erase_diff/configs/train_config.yaml \
--train_method xattn \
--alpha 0.2 \
--epochs 10 \
--lr 1e-4 \
--devices 0,1 \
--raw_dataset_dir /path/to/raw_dataset \
--output_dir outputs/experiment_1
```

**Explanation of the Example**

* --config_path: Specifies the YAML configuration file to load default values.
* --train_method: Overrides the training method ("xattn").
* --alpha: Sets the guidance strength for the starting image to 0.2.
* --epochs: Increases the number of training epochs to 10.
* --lr: Updates the learning rate to 1e-4.
* --devices: Specifies the GPUs (e.g., device 0 and 1) for training.
* --raw_dataset_dir: Changes the raw dataset directory.
* --output_dir: Sets a custom output directory for this run.


**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.


### Directory Structure

- `algorithm.py`: Implementation of the EraseDiffAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the EraseDiffModel class.
- `scripts/train.py`: Script to train the EraseDiff algorithm.
- `trainer.py`: Implementation of the EraseDiffTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class

---

### Description of Arguments in train_config.yaml

**Training Parameters**

* train_method: Specifies the method of training for concept erasure.

    * Choices: ["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"]
    * Example: "xattn"

* alpha: Guidance strength for the starting image during training.

    * Type: float
    * Example: 0.1

* epochs: Number of epochs to train the model.

    * Type: int
    * Example: 1

* K_steps: Number of K optimization steps during training.

    * Type: int
    * Example: 2

* lr: Learning rate used for the optimizer during training.

    * Type: float
    * Example: 5e-5

**Model Configuration**

* model_config_path: File path to the Stable Diffusion model configuration YAML file.

    * type: str
    * Example: "/path/to/model_config.yaml"

* ckpt_path: File path to the checkpoint of the Stable Diffusion model.

    * Type: str
    * Example: "/path/to/model_checkpoint.ckpt"


**Dataset Directories**

* raw_dataset_dir: Directory containing the raw dataset categorized by themes or classes.

    * Type: str
    * Example: "/path/to/raw_dataset"

* processed_dataset_dir: Directory to save the processed dataset.

    * Type: str
    * Example: "/path/to/processed_dataset"

* dataset_type: Specifies the dataset type for the training process.

    * Choices: ["unlearncanvas", "i2p"]
    * Example: "unlearncanvas"

* template: Type of template to use during training.

    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* template_name: Name of the specific concept or style to be erased.

    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"


**Output Configurations**

* output_dir: Directory where the fine-tuned models and results will be saved.

    * Type: str
    * Example: "outputs/erase_diff/finetuned_models"

* separator: String separator used to train multiple words separately, if applicable.

    * Type: str or null
    * Example: null

**Sampling and Image Configurations**

* image_size: Size of the training images (height and width in pixels).

    * Type: int
    * Example: 512

* interpolation: Interpolation method used for image resizing.

    * Choices: ["bilinear", "bicubic", "lanczos"]
    * Example: "bicubic"

* ddim_steps: Number of DDIM inference steps during training.

    * Type: int
    * Example: 50

* ddim_eta: DDIM eta parameter for stochasticity during sampling.

    * Type: float
    * Example: 0.0

**Device Configuration**

* devices: Specifies the CUDA devices to be used for training (comma-separated).

    * Type: str
    * Example: "0"


**Additional Flags**

* use_sample: Flag to indicate whether a sample dataset should be used for training.

    * Type: bool
    * Example: True

* num_workers: Number of worker threads for data loading.

    * Type: int
    * Example: 4

* pin_memory: Flag to enable pinning memory during data loading for faster GPU transfers.

    * Type: bool
    * Example: true


