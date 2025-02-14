### Train Config
```python

class SaliencyUnlearningConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Model configuration
        self.alpha = 0.1  # Alpha value for training
        self.epochs = 1  # Number of epochs for training
        self.train_method = (
            "xattn"  # Attention method: ["noxattn", "selfattn", "xattn", "full"]
        )
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"  # Path to the checkpoint
        self.model_config_path = current_dir / "model_config.yaml"

        # Dataset directories
        self.raw_dataset_dir = (
            "data/quick-canvas-dataset/sample"  # Path to the raw dataset
        )
        self.processed_dataset_dir = (
            "mu/algorithms/saliency_unlearning/data"  # Path to the processed dataset
        )
        self.dataset_type = "unlearncanvas"  # Type of the dataset
        self.template = "style"  # Template type for training
        self.template_name = "Abstractionism"  # Name of the template

        # Directory Configuration
        self.output_dir = "outputs/saliency_unlearning/finetuned_models"  # Directory for output models
        self.mask_path = (
            "outputs/saliency_unlearning/masks/0.5.pt"  # Path to the mask file
        )

        # Training configuration
        self.devices = "0"  # CUDA devices for training (comma-separated)
        self.use_sample = True  # Whether to use a sample dataset for training

        # Guidance and training parameters
        self.start_guidance = 0.5  # Start guidance for training
        self.negative_guidance = 0.5  # Negative guidance for training
        self.ddim_steps = 50  # Number of DDIM steps for sampling

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
```


### Model Config
```yaml

# Model Configuration
alpha: 0.1
epochs: 1
train_method: "xattn"  # Choices: ["noxattn", "selfattn", "xattn", "full" ]
ckpt_path: "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion
model_config_path: "mu/algorithms/saliency_unlearning/configs/model_config.yaml"  # Config path for Stable Diffusion

# Dataset directories
raw_dataset_dir: "data/quick-canvas-dataset/sample"
processed_dataset_dir: "mu/algorithms/saliency_unlearning/data"
dataset_type : "unlearncanvas"
template : "style"
template_name : "Abstractionism"

# Directory Configuration
output_dir: "outputs/saliency_unlearning/finetuned_models"  # Output directory to save results
mask_path: "outputs/saliency_unlearning/masks/0.5.pt"  # Output directory to save results

# Training Configuration
devices: "0"  # CUDA devices to train on (comma-separated)
use_sample: true


start_guidance: 0.5
negative_guidance: 0.5
ddim_steps: 50
```

### Mask Config
```
# Model Configuration
c_guidance: 7.5
batch_size: 4
num_timesteps: 1000
image_size: 512

model_config_path: "mu/algorithms/saliency_unlearning/configs/model_config.yaml"
# ckpt_path: "models/compvis/style50/compvis.ckpt"  # Checkpoint path for Stable Diffusion
ckpt_path: "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/compvis/style50/compvis.ckpt" 

# Dataset directories
# raw_dataset_dir: "data/quick-canvas-dataset/sample"
raw_dataset_dir: "/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample"
processed_dataset_dir: "mu/algorithms/saliency_unlearning/data"
dataset_type : "unlearncanvas"
template : "style"
template_name : "Abstractionism"
threshold : 0.5

# Directory Configuration
output_dir: "outputs/saliency_unlearning/masks"  # Output directory to save results

# Training Configuration
lr: 0.00001
devices: "0"  # CUDA devices to train on (comma-separated)
use_sample: true
```


### Evaluation config

```python
# mu/algorithms/saliency_unlearning/configs/evaluation_config.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class SaliencyUnlearningEvaluationConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.model_config_path = current_dir/"model_config.yaml"  # path to model config
        self.ckpt_path = "outputs/saliency_unlearning/finetuned_models/saliency_unlearning_Abstractionism_model.pth"  # path to finetuned model checkpoint
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
        self.sampler_output_dir = "outputs/eval_results/mu_results/saliency_unlearning/"  # directory to save sampler outputs
        self.seed_list = ["188"]  # list of seeds for evaluation
        self.classification_model = "vit_large_patch16_224"  # classification model for evaluation
        self.eval_output_dir = "outputs/eval_results/mu_results/saliency_unlearning/"  # directory to save evaluation results
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


saliency_unlearning_evaluation_config = SaliencyUnlearningEvaluationConfig()
```