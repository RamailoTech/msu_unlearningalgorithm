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

### Description of configs used to generate mask:


**Model Configuration**

These parameters specify settings for the Stable Diffusion model and guidance configurations.

* c_guidance: Guidance scale used during loss computation in the model. Higher values may emphasize certain features in mask generation.
    
    * Type: float
    * Example: 7.5

* batch_size: Number of images processed in a single batch.

    * Type: int
    * Example: 4

* ckpt_path: Path to the model checkpoint file for Stable Diffusion.

    * Type: str
    * Example: /path/to/compvis.ckpt

* model_config_path: Path to the model configuration YAML file for Stable Diffusion.

    * Type: str
    * Example: /path/to/model_config.yaml

* num_timesteps: Number of timesteps used in the diffusion process.

    * Type: int
    * Example: 1000

* image_size: Size of the input images used for training and mask generation (in pixels).

    * Type: int
    * Example: 512


**Dataset Configuration**

These parameters define the dataset paths and settings for mask generation.

* raw_dataset_dir: Path to the directory containing the original dataset, organized by themes and classes.

    * Type: str
    * Example: /path/to/raw/dataset

* processed_dataset_dir: Path to the directory where processed datasets will be saved after mask generation.

    * Type: str
    * Example: /path/to/processed/dataset

* dataset_type: Type of dataset being used.

    * Choices: unlearncanvas, i2p
    * Type: str
    * Example: i2p

* template: Type of template for mask generation.

    * Choices: object, style, i2p
    * Type: str
    * Example: style

* template_name: Specific template name for the mask generation process.

    * Example Choices: self-harm, Abstractionism
    * Type: str
    * Example: Abstractionism

* threshold: Threshold value for mask generation to filter salient regions.

    * Type: float
    * Example: 0.5

**Output Configuration**

These parameters specify the directory where the results are saved.

* output_dir: Directory where the generated masks will be saved.

    * Type: str
    * Example: outputs/saliency_unlearning/masks


**Training Configuration**

These parameters control the training process for mask generation.

* lr: Learning rate used for training the masking algorithm.

    * Type: float
    * Example: 0.00001

* devices: CUDA devices used for training, specified as a comma-separated list.

    * Type: str
    * Example: 0

* use_sample: Flag indicating whether to use a sample dataset for training and mask generation.

    * Type: bool
    * Example: True


### Description of Arguments used to train saliency unlearning.

The following configs are used to fine-tune the Stable Diffusion model to perform saliency-based unlearning. This script relies on a configuration class `SaliencyUnlearningConfig`  and supports additional runtime arguments for further customization. Below is a detailed description of each argument:

**General Arguments**

* alpha: Guidance scale used to balance the loss components during training.
    
    * Type: float
    * Example: 0.1

* epochs: Number of epochs to train the model.
    
    * Type: int
    * Example: 5

* train_method: Specifies the training method or strategy to be used.

    * Choices: noxattn, selfattn, xattn, full, notime, xlayer, selflayer
    * Type: str
    * Example: noxattn

* model_config_path: Path to the model configuration YAML file for Stable Diffusion.
    
    * Type: str
    * Example: 'mu/algorithms/saliency_unlearning/configs/model_config.yaml'


**Dataset Arguments**

* raw_dataset_dir: Path to the directory containing the raw dataset, organized by themes and classes.

    * Type: str
    * Example: 'path/raw_dataset/'

* processed_dataset_dir: Path to the directory where the processed dataset will be saved.

    * Type: str
    * Example: 'path/processed_dataset_dir'

* dataset_type: Specifies the type of dataset to use for training. Use `generic` as type if you want to use your own dataset.

    * Choices: unlearncanvas, i2p, generic
    * Type: str
    * Example: i2p

* template: Specifies the template type for training.

    * Choices: object, style, i2p
    * Type: str
    * Example: style

* template_name: Name of the specific template used for training.

    * Example Choices: self-harm, Abstractionism
    * Type: str
    * Example: Abstractionism


**Output Arguments**

* output_dir: Directory where the fine-tuned model and training outputs will be saved.

    * Type: str
    * Example: 'output/folder_name'

* mask_path: Path to the saliency mask file used during training.

    * Type: str
    * Example: 





