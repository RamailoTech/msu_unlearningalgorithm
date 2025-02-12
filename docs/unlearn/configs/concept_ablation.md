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


### Evaluation config


```python
#mu/algorithms/concept_ablation/configs/evaluation_config.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class ConceptAblationEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.model_config_path = current_dir/"model_config.yaml"  # path to model config
        self.ckpt_path = "outputs/concept_ablation/finetuned_models/checkpoints/last-v2.ckpt"  # path to finetuned model checkpoint
        self.classifier_ckpt_path = "models/classifier_ckpt_path/style50_cls.pth"  # path to classifier checkpoint
        self.forget_theme = "Bricks"  # theme to forget
        self.cfg_text = 9.0  # classifier-free guidance scale
        self.devices = "0"  # GPU device ID
        self.seed = 188  # random seed
        self.task = "class"  # task type
        self.ddim_steps = 100  # number of DDIM steps
        self.image_height = 512  # height of the image
        self.image_width = 512  # width of the image
        self.ddim_eta = 0.0  # DDIM eta parameter
        self.sampler_output_dir = "outputs/eval_results/mu_results/ca/"  # directory to save sampler outputs
        self.seed_list = ["188"]  # list of seeds for evaluation
        self.classification_model = "vit_large_patch16_224"  # classification model for evaluation
        self.eval_output_dir = "outputs/eval_results/mu_results/ca/"  # directory to save evaluation results
        self.reference_dir = "data/quick-canvas-dataset/sample/"  # path to the original dataset
        self.multiprocessing = False  # whether to use multiprocessing

        # Override defaults with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"Model config file {self.model_config_path} does not exist.")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if not os.path.exists(self.classifier_ckpt_path):
            raise FileNotFoundError(f"Classifier checkpoint file {self.classifier_ckpt_path} does not exist.")
        if not os.path.exists(self.reference_dir):
            raise FileNotFoundError(f"Reference directory {self.reference_dir} does not exist.")
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir)
        if not os.path.exists(self.sampler_output_dir):
            os.makedirs(self.sampler_output_dir)

        if self.cfg_text <= 0:
            raise ValueError("Classifier-free guidance scale (cfg_text) should be positive.")
        if self.ddim_steps <= 0:
            raise ValueError("DDIM steps should be a positive integer.")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("Image height and width should be positive.")
        if self.task not in ["class", "other_task"]:  # Add other valid tasks if needed
            raise ValueError("Invalid task type.")


# Example usage
concept_ablation_evaluation_config = ConceptAblationEvaluationConfig()
```