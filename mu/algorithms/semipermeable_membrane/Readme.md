# Semi Permeable Membrane Algorithm for Machine Unlearning

This repository provides an implementation of the semipermeable membrane algorithm for machine unlearning in Stable Diffusion models. The semipermeable membrane algorithm allows you to remove specific concepts or styles from a pre-trained model without retraining it from scratch.

## Installation

### Create the Conda Environment

First, create and activate the Conda environment using the provided `environment.yaml` file:

```bash
conda env create -f mu/algorithms/semipermeable_membrane/environment.yaml -n mu_semipermeable_membrane
```

```bash
conda activate mu_semipermeable_membrane
```

### Download models

To download [models](https://huggingface.co/nebulaanish/unlearn_models/tree/main), use the following commands <br>

1. Compvis (Size 3.84 GB):

    * Make it executable:

        `chmod +x scripts/download_models.sh`

    * Run the script:
        ```scripts/download_models.sh compvis```

2. Diffuser (24.1 GB): 

    * Make it executable:

        `chmod +x scripts/download_models.sh`

    * Run the script: 
        ```scripts/download_models.sh diffuser```

**Notes:**

1. The script ensures that directories are automatically created if they don’t exist.
2. The downloaded ZIP file will be extracted to the respective folder, and the ZIP file will be removed after extraction.


**Verify Downloads**

After downloading, you can verify the extracted files in their respective directories:

`ls -lh ../models/compvis/`

`ls -lh ../models/diffuser/`

### Download datasets

1. Download unlearn canvas dataset:

    * Make it executable:

        `chmod +x scripts/download_quick_canvas_dataset.sh`

    * Download the sample dataset (smaller size):

        `scripts/download_quick_canvas_dataset.sh sample`

    * Download the full dataset:

        `scripts/download_quick_canvas_dataset.sh full`

2. Download the i2p dataset

    * Make it executable:

        `chmod +x scripts/download_i2p_dataset.sh`

    * Download the sample dataset (smaller size):

        `scripts/download_i2p_dataset.sh sample`

    * Download the full dataset:

        `scripts/download_i2p_dataset.sh full`

**Notes:**

1. The script automatically creates the required directories if they don't exist.
2. Ensure curl and unzip are installed on your system.

**Verify the Downloaded files**

After downloading, verify that the datasets have been correctly extracted:

`ls -lh ./data/i2p-dataset/sample/`

`ls -lh ./data/quick-canvas-dataset/sample/`


### Description of Arguments in train_config.yaml

**pretrained_model**

* ckpt_path: File path to the pretrained model's checkpoint file.

* v2: Boolean indicating whether the pretrained model is version 2 or not.

* v_pred: Boolean to enable/disable "v-prediction" mode for diffusion models.

* clip_skip: Number of CLIP layers to skip during inference.

**network**

* rank: Rank of the low-rank adaptation network.

* alpha: Scaling factor for the network during training.


**train**

* precision: Numerical precision to use during training (e.g., float32 or float16).

* noise_scheduler: Type of noise scheduler to use in the training loop (e.g., ddim).

* iterations: Number of training iterations.

* batch_size: Batch size for training.

* lr: Learning rate for the training optimizer.

* unet_lr: Learning rate for the U-Net model.

* text_encoder_lr: Learning rate for the text encoder.

* optimizer_type: Optimizer to use for training (e.g., AdamW8bit).

* lr_scheduler: Learning rate scheduler to apply during training.

* lr_warmup_steps: Number of steps for linear warmup of the learning rate.

* lr_scheduler_num_cycles: Number of cycles for a cosine-with-restarts scheduler.

* max_denoising_steps: Maximum denoising steps to use during training.

**save**

* per_steps: Frequency of saving the model (in steps).

* precision: Numerical precision for saved model weights


**other**

* use_xformers: Boolean to enable xformers memory-efficient attention.

* wandb_project and wandb_run

* Configuration for tracking the training progress using Weights & Biases.

* wandb_project: Project name in W&B.

* wandb_run: Specific run name in the W&B dashboard.

**use_sample**

* Boolean to indicate whether to use the sample dataset for training.

**dataset_type**

* Type of dataset to use, options are unlearncanvas or i2p.

**template**
* Specifies the template type, choices are:
    * object: Focus on specific objects.
    * style: Focus on artistic styles.
    * i2p: Intermediate style processing.

**template_name**

* Name of the template, choices are:
    * self-harm
    * Abstractionism

**prompt**

* target: Target template or concept to guide training (references template_name).

* positive: Positive prompt based on the template.

* unconditional: Unconditional prompt text.

* neutral: Neutral prompt text.

* action: Specifies the action applied to the prompt (e.g., erase_with_la).

* guidance_scale: Guidance scale for classifier-free guidance.

* resolution: Image resolution for training.

* batch_size: Batch size for generating prompts.

* dynamic_resolution: Boolean to allow dynamic resolution.

* la_strength: Strength of local adaptation.

* sampling_batch_size: Batch size for sampling images.

**devices**

* CUDA devices to use for training (specified as a comma-separated list, e.g., "0,1").

**output_dir**

* Directory to save the fine-tuned model and other outputs.

**verbose**

* Boolean flag for verbose logging during training.

## Usage

To train the Semi Permeable Membrane algorithm to unlearn a specific concept or style from the Stable Diffusion model, use the `train.py` script located in the `scripts` directory.

### Example Command

```bash
python -m mu.algorithms.semipermeable_membrane.scripts.train \
--config_path mu/algorithms/semipermeable_membrane/config/train_config.yaml
```

**Running the Script in Offline Mode**

```bash
WANDB_MODE=offline python -m mu.algorithms.semipermeable_membrane.scripts.train \
--config_path mu/algorithms/semipermeable_membrane/config/train_config.yaml
```



**Passing Arguments via the Command Line**

The `train.py` script allows you to override configuration parameters specified in the `train_config.yaml` file by passing them directly as arguments during runtime. This can be useful for quick experimentation without modifying the configuration file.



**Example Usage with Command-Line Arguments**

```bash
python -m mu.algorithms.semipermeable_membrane.scripts.train \
--config_path mu/algorithms/semipermeable_membrane/config/train_config.yaml \
--dataset_type unlearncanvas \
--template object \
--template_name self-harm \
--devices 0,1 \
--output_dir outputs/experiment_1 \
--use_sample
```


**Explanation of the Example**

* --config_path: Specifies the YAML configuration file to load default values.

* --dataset_type: Defines the dataset type (e.g., unlearncanvas or i2p).

* --template: Specifies the template to use (e.g., object, style, or i2p).

* --template_name: The specific name for the template being used (e.g., self-harm, Abstractionism).

* --devices: Comma-separated list of CUDA devices to use for training (e.g., 0,1 for using GPU 0 and GPU 1).

* --output_dir: Sets a custom output directory for the results of this run.

* --use_sample: Specifies whether to use a sample dataset for training.


**How It Works** 
* Default Values: The script first loads default values from the YAML file specified by --config_path.

* Command-Line Overrides: Any arguments passed on the command line will override the corresponding keys in the YAML configuration file.

* Final Configuration: The script merges the YAML file and command-line arguments into a single configuration dictionary and uses it for training.

## Directory Structure

- `algorithm.py`: Implementation of the Semi Permeable MembraneAlgorithm class.
- `configs/`: Contains configuration files for training and generation.
- `model.py`: Implementation of the Semi Permeable MembraneModel class.
- `scripts/train.py`: Script to train the Semi Permeable Membrane algorithm.
- `trainer.py`: Implementation of the Semi Permeable MembraneTrainer class.
- `utils.py`: Utility functions used in the project.
- `data_handler.py` : Implementation of DataHandler class


