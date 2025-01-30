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