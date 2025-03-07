### Train Config
```python
class ConceptAblationConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Training parameters
        self.seed = 23  # Seed for random number generators
        self.scale_lr = True  # Flag to scale the learning rate
        self.caption_target = "Abstractionism Style"  # Caption target for the training
        self.regularization = True  # Whether to apply regularization
        self.n_samples = 10  # Number of samples to generate
        self.train_size = 200  # Number of training samples
        self.base_lr = 2.0e-06  # Base learning rate

        # Model configuration
        self.config_path = current_dir / "train_config.yaml"
        self.model_config_path = (
            current_dir / "model_config.yaml"
        )  # Path to model config
        self.ckpt_path = (
            "models/compvis/style50/compvis.ckpt"  # Path to model checkpoint
        )

        # Dataset directories
        self.raw_dataset_dir = (
            "data/quick-canvas-dataset/sample"  # Raw dataset directory
        )
        self.processed_dataset_dir = (
            "mu/algorithms/concept_ablation/data"  # Processed dataset directory
        )
        self.dataset_type = "unlearncanvas"  # Dataset type
        self.template = "style"  # Template used for training
        self.template_name = "Abstractionism"  # Template name

        # Learning rate for training
        self.lr = 5e-5  # Learning rate

        # Output directory for saving models
        self.output_dir = (
            "outputs/concept_ablation/finetuned_models"  # Output directory for results
        )

        # Device configuration
        self.devices = "0"  # CUDA devices (comma-separated)

        # Additional flags
        self.use_sample = True  # Whether to use the sample dataset for training

        # Data configuration
        self.data = {
            "target": "mu.algorithms.concept_ablation.data_handler.ConceptAblationDataHandler",
            "params": {
                "batch_size": 1,  # Batch size for training
                "num_workers": 1,  # Number of workers for loading data
                "wrap": False,  # Whether to wrap the dataset
                "train": {
                    "target": "mu.algorithms.concept_ablation.src.finetune_data.MaskBase",
                    "params": {"size": 512},  # Image size for the training set
                },
                "train2": {
                    "target": "mu.algorithms.concept_ablation.src.finetune_data.MaskBase",
                    "params": {"size": 512},  # Image size for the second training set
                },
            },
        }

        # Lightning configuration
        self.lightning = {
            "callbacks": {
                "image_logger": {
                    "target": "mu.algorithms.concept_ablation.callbacks.ImageLogger",
                    "params": {
                        "batch_frequency": 20000,  # Frequency to log images
                        "save_freq": 10000,  # Frequency to save images
                        "max_images": 8,  # Maximum number of images to log
                        "increase_log_steps": False,  # Whether to increase the logging steps
                    },
                }
            },
            "modelcheckpoint": {
                "params": {
                    "every_n_train_steps": 10000  # Save the model every N training steps
                }
            },
            "trainer": {"max_steps": 2000},  # Maximum number of training steps
        }

        self.prompts = "mu/algorithms/concept_ablation/data/anchor_prompts/finetune_prompts/sd_prompt_Architectures_sample.txt"

```

### Model Config
```yaml
# Training parameters
seed : 23 
scale_lr : True 
caption_target : "Abstractionism Style"
regularization : True 
n_samples : 10 
train_size : 200
base_lr : 2.0e-06

# Model configuration
model_config_path: "mu/algorithms/concept_ablation/configs/model_config.yaml"  # Config path for Stable Diffusion
ckpt_path: "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion

# Dataset directories
raw_dataset_dir: "data/quick-canvas-dataset/sample"
processed_dataset_dir: "mu/algorithms/concept_ablation/data"
dataset_type : "unlearncanvas"
template : "style"
template_name : "Abstractionism"

lr: 5e-5 
# Output configurations
output_dir: "outputs/concept_ablation/finetuned_models"  # Output directory to save results

# Sampling and image configurations

# Device configuration
devices: "0,"  # CUDA devices to train on (comma-separated)

# Additional flags
use_sample: True  # Use the sample dataset for training

data:
  target: mu.algorithms.concept_ablation.data_handler.ConceptAblationDataHandler
  params:
    batch_size: 4
    num_workers: 4
    wrap: false
    train:
      target: mu.algorithms.concept_ablation.src.finetune_data.MaskBase
      params:
        size: 512
    train2:
      target: mu.algorithms.concept_ablation.src.finetune_data.MaskBase
      params:
        size: 512


lightning:
  callbacks:
    image_logger:
      target: mu.algorithms.concept_ablation.callbacks.ImageLogger
      params:
        batch_frequency: 20000
        save_freq: 10000
        max_images: 8
        increase_log_steps: False
  modelcheckpoint:
    params:
      every_n_train_steps: 10000

  trainer:
    max_steps: 2000
```

## Configuration File description

### Training Parameters

* **seed:** Random seed for reproducibility.
    * Type: int
    * Example: 23

* **scale_lr:** Whether to scale the base learning rate.
    * Type: bool
    * Example: True

* **caption_target:** Target style to remove.
    * Type: str
    * Example: "Abstractionism Style"

* **regularization:** Adds regularization loss during training.
    * Type: bool
    * Example: True

* **n_samples:** Number of batch sizes for image generation.
    * Type: int
    * Example: 10

* **train_size:** Number of generated images for training.
    * Type: int
    * Example: 1000

* **base_lr:** Learning rate for the optimizer.
    * Type: float
    * Example: 2.0e-06

### Model Configuration

* **model_config_path:** Path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/model_config.yaml"

* **ckpt_path:** Path to the Stable Diffusion model checkpoint.
    * Type: str
    * Example: "/path/to/compvis.ckpt"

### Dataset Directories

* **raw_dataset_dir:** Directory containing the raw dataset categorized by themes or classes.
    * Type: str
    * Example: "/path/to/raw_dataset"

* **processed_dataset_dir:** Directory to save the processed dataset.
    * Type: str
    * Example: "/path/to/processed_dataset"

* **dataset_type:** Specifies the dataset type for training. Use `generic` as type if you want to use your own dataset.
    * Choices: ["unlearncanvas", "i2p","generic"]
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
    * Example: "outputs/concept_ablation/finetuned_models"

### Device Configuration

* **devices:** CUDA devices for training (comma-separated).
    * Type: str
    * Example: "0"






