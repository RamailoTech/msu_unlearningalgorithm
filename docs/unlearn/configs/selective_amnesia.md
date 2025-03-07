### Train config

```python
import os
from mu.core.base_config import BaseConfig
from pathlib import Path

current_dir = Path(__file__).parent


class SelectiveAmnesiaConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Training parameters
        self.seed = 23  # Random seed
        self.scale_lr = True  # Flag for scaling learning rate

        # Model configuration
        self.config_path = current_dir / "train_config.yaml"
        self.model_config_path = (
            current_dir / "model_config.yaml"
        )  # Config path for model
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion
        self.full_fisher_dict_pkl_path = "mu/algorithms/selective_amnesia/data/full_fisher_dict.pkl"  # Path for Fisher dict

        # Dataset directories
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/selective_amnesia/data"
        self.dataset_type = (
            "unlearncanvas"  # Dataset type (choices: unlearncanvas, i2p)
        )
        self.template = "style"  # Template to use
        self.template_name = "Abstractionism"  # Template name
        self.replay_prompt_path = "mu/algorithms/selective_amnesia/data/fim_prompts_sample.txt"  # Path for replay prompts

        # Output configurations
        self.output_dir = "outputs/selective_amnesia/finetuned_models"  # Output directory to save results

        # Device configuration
        self.devices = "0,"  # CUDA devices (comma-separated)

        # Additional flags
        self.use_sample = True  # Use sample dataset for training

        # Data configuration
        self.data = {
            "target": "mu.algorithms.selective_amnesia.data_handler.SelectiveAmnesiaDataHandler",
            "params": {
                "train_batch_size": 4,
                "val_batch_size": 6,
                "num_workers": 1,
                "num_val_workers": 0,  # Avoid val dataloader issue
                "train": {
                    "target": "stable_diffusion.ldm.data.ForgettingDataset",
                    "params": {
                        "forget_prompt": "An image in Artist_Sketch style",
                        "forget_dataset_path": "./q_dist/photo_style",
                    },
                },
                "validation": {
                    "target": "stable_diffusion.ldm.data.VisualizationDataset",
                    "params": {
                        "output_size": 512,
                        "n_gpus": 1,  # Number of GPUs for validation
                    },
                },
            },
        }

        # Lightning configuration
        self.lightning = {
            "find_unused_parameters": False,
            "modelcheckpoint": {
                "params": {"every_n_epochs": 0, "save_top_k": 0, "monitor": None}
            },
            "callbacks": {
                "image_logger": {
                    "target": "mu.algorithms.selective_amnesia.callbacks.ImageLogger",
                    "params": {
                        "batch_frequency": 1,
                        "max_images": 999,
                        "increase_log_steps": False,
                        "log_first_step": False,
                        "log_all_val": True,
                        "clamp": True,
                        "log_images_kwargs": {
                            "ddim_eta": 0,
                            "ddim_steps": 50,
                            "use_ema_scope": True,
                            "inpaint": False,
                            "plot_progressive_rows": False,
                            "plot_diffusion_rows": False,
                            "N": 6,  # Number of validation prompts
                            "unconditional_guidance_scale": 7.5,
                            "unconditional_guidance_label": [""],
                        },
                    },
                }
            },
            "trainer": {
                "benchmark": True,
                "num_sanity_val_steps": 0,
                "max_epochs": 50,  # Modify epochs here!
                "check_val_every_n_epoch": 10,
            },
        }

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # Check if necessary directories exist
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory {self.processed_dataset_dir} does not exist."
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Check if model and checkpoint files exist
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model config file {self.model_config_path} does not exist."
            )
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if not os.path.exists(self.full_fisher_dict_pkl_path):
            raise FileNotFoundError(
                f"Fisher dictionary file {self.full_fisher_dict_pkl_path} does not exist."
            )

        # Check if replay prompts file exists
        if not os.path.exists(self.replay_prompt_path):
            raise FileNotFoundError(
                f"Replay prompt file {self.replay_prompt_path} does not exist."
            )

        # Validate dataset type
        if self.dataset_type not in ["unlearncanvas", "i2p"]:
            raise ValueError(
                f"Invalid dataset type {self.dataset_type}. Choose from ['unlearncanvas', 'i2p']"
            )

        # Validate batch sizes
        if self.data["params"]["train_batch_size"] <= 0:
            raise ValueError(f"train_batch_size should be a positive integer.")
        if self.data["params"]["val_batch_size"] <= 0:
            raise ValueError(f"val_batch_size should be a positive integer.")

        # Validate lightning trainer max_epochs
        if self.lightning["trainer"]["max_epochs"] <= 0:
            raise ValueError(f"max_epochs should be a positive integer.")


selective_amnesia_config_quick_canvas = SelectiveAmnesiaConfig()
selective_amnesia_config_quick_canvas.dataset_type = "unlearncanvas"
selective_amnesia_config_quick_canvas.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

selective_amnesia_config_i2p = SelectiveAmnesiaConfig()
selective_amnesia_config_i2p.dataset_type = "i2p"
selective_amnesia_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
```


### Train config yaml file

```bash
# Training parameters
seed : 23 
scale_lr : True 

# Model configuration
model_config_path: "mu/algorithms/selective_amnesia/configs/model_config.yaml"
ckpt_path: "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion
full_fisher_dict_pkl_path : "mu/algorithms/selective_amnesia/data/full_fisher_dict.pkl"

# Dataset directories
raw_dataset_dir: "data/quick-canvas-dataset/sample"
processed_dataset_dir: "mu/algorithms/selective_amnesia/data"
dataset_type : "unlearncanvas"
template : "style"
template_name : "Abstractionism"
replay_prompt_path: "mu/algorithms/selective_amnesia/data/fim_prompts_sample.txt"


# Output configurations
output_dir: "outputs/selective_amnesia/finetuned_models"  # Output directory to save results

# Sampling and image configurations

# Device configuration
devices: "0,"  # CUDA devices to train on (comma-separated)

# Additional flags
use_sample: True  # Use the sample dataset for training

data:
  target: mu.algorithms.selective_amnesia.data_handler.SelectiveAmnesiaDataHandler
  params:
    train_batch_size: 4
    val_batch_size: 6
    num_workers: 4
    num_val_workers: 0 # Avoid a weird val dataloader issue (keep unchanged)
    train:
      target: stable_diffusion.ldm.data.ForgettingDataset
      params:
        forget_prompt: An image in Artist_Sketch style
        forget_dataset_path: ./q_dist/photo_style
    validation:
      target: stable_diffusion.ldm.data.VisualizationDataset
      params:
        output_size: 512
        n_gpus: 1 # CHANGE THIS TO NUMBER OF GPUS! small hack to sure we see all our logging samples

lightning:
  find_unused_parameters: False

  modelcheckpoint:
    params:
      every_n_epochs: 0
      save_top_k: 0
      monitor: null

  callbacks:
    image_logger:
      target: mu.algorithms.selective_amnesia.callbacks.ImageLogger
      params:
        batch_frequency: 1
        max_images: 999
        increase_log_steps: False
        log_first_step: False
        log_all_val: True
        clamp: True
        log_images_kwargs:
          ddim_eta: 0
          ddim_steps: 50
          use_ema_scope: True
          inpaint: False
          plot_progressive_rows: False
          plot_diffusion_rows: False
          N: 6 # keep this the same as number of validation prompts!
          unconditional_guidance_scale: 7.5
          unconditional_guidance_label: [""]

  trainer:
    benchmark: True
    num_sanity_val_steps: 0
    max_epochs: 50 # modify epochs here!
    check_val_every_n_epoch: 10
```

## Configuration File description

### Training Parameters

* **seed:** Random seed for reproducibility.
    * Type: int
    * Example: 23

* **scale_lr:** Whether to scale the base learning rate.
    * Type: bool
    * Example: True

### Model Configuration

* **model_config_path:** Path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/model_config.yaml"

* **ckpt_path:** Path to the Stable Diffusion model checkpoint.
    * Type: str
    * Example: "/path/to/compvis.ckpt"

* **full_fisher_dict_pkl_path:** Path to the full fisher dict pkl file
    * Type: str
    * Example: "full_fisher_dict.pkl"

### Dataset Directories

* **raw_dataset_dir:** Directory containing the raw dataset categorized by themes or classes.
    * Type: str
    * Example: "/path/to/raw_dataset"

* **processed_dataset_dir:** Directory to save the processed dataset.
    * Type: str
    * Example: "/path/to/processed_dataset"

* **dataset_type:** Specifies the dataset type for training. Use `generic` as type if you want to use your own dataset.
    * Choices: ["unlearncanvas", "i2p", "generic"]
    * Example: "unlearncanvas"

* **template:** Type of template to use during training.
    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* **template_name:** Name of the concept or style to erase.
    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"

### Output Configurations

* **output_dir:** Directory to save fine-tuned models and results.
    * Type: str
    * Example: "outputs/selective_amnesia/finetuned_models"

### Device Configuration

* **devices:** CUDA devices for training (comma-separated).
    * Type: str
    * Example: "0"

### Data Parameters

* **train_batch_size:** Batch size for training.
    * Type: int
    * Example: 4

* **val_batch_size:** Batch size for validation.
    * Type: int
    * Example: 6

* **num_workers:** Number of worker threads for data loading.
    * Type: int
    * Example: 4

* **forget_prompt:** Prompt to specify the style or concept to forget.
    * Type: str
    * Example: "An image in Artist_Sketch style"

### Lightning Configuration

* **max_epochs:** Maximum number of epochs for training.
    * Type: int
    * Example: 50

* **callbacks:**
    * **batch_frequency:** Frequency for logging image batches.
        * Type: int
        * Example: 1

    * **max_images:** Maximum number of images to log.
        * Type: int
        * Example: 999

---

