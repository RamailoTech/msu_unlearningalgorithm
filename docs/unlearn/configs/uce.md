### Train Config
```python
class UnifiedConceptEditingConfig(BaseConfig):
    def __init__(self, **kwargs):
        # Training configuration
        self.train_method = "full"  # Options: full, partial
        self.alpha = 0.1  # Guidance factor for training
        self.epochs = 1  # Number of epochs
        self.lr = 5e-5  # Learning rate

        # Model configuration
        self.ckpt_path = "models/diffuser/style50"  # Path to model checkpoint

        # Output configuration
        self.output_dir = (
            "outputs/uce/finetuned_models"  # Directory to save finetuned models
        )
        self.dataset_type = "unlearncanvas"  # Type of dataset to be used
        self.template = "style"  # Template for training
        self.template_name = "Abstractionism"  # Name of the template

        # Device configuration
        self.devices = "0"  # CUDA devices to train on (comma-separated)

        # Additional flags
        self.use_sample = True  # Whether to use the sample dataset

        # Editing-specific configuration
        self.guided_concepts = (
            "A Elephant image"  # Comma-separated string of guided concepts
        )
        self.technique = (
            "replace"  # Technique for editing (Options: "replace", "tensor")
        )

        # Parameters for the editing technique
        self.preserve_scale = 0.1  # Scale for preserving the concept (float)
        self.preserve_number = (
            None  # Number of concepts to preserve (int, None for all)
        )
        self.erase_scale = 1  # Scale for erasing
        self.lamb = 0.1  # Regularization weight for loss
        self.add_prompts = False  # Whether to add additional prompts

        # Preserver concepts (comma-separated if multiple)
        self.preserver_concepts = (
            "A Lion image"  # Comma-separated string of preserver concepts
        )

        # Base model used for editing
        self.base = "stable-diffusion-v1-4"  # Base version of Stable Diffusion
```

### Description of Arguments in train_config.yaml
**Training Parameters**

* **train_method**: Specifies the method of training for concept erasure.
    * Choices: ["full", "partial"]
    * Example: "full"

* **alpha**: Guidance strength for the starting image during training.
    * Type: float
    * Example: 0.1

* **epochs**: Number of epochs to train the model.
    * Type: int
    * Example: 10

* **lr**: Learning rate used for the optimizer during training.
    * Type: float
    * Example: 5e-5


**Model Configuration**
* **ckpt_path**: File path to the checkpoint of the Stable Diffusion model.
    * Type: str
    * Example: "/path/to/model_checkpoint.ckpt"

* **config_path**: File path to the Stable Diffusion model configuration YAML file.
    * Type: str
    * Example: "/path/to/config.yaml"

**Dataset Directories**

* **dataset_type**: Specifies the dataset type for the training process. Use `generic` as type if you want to use your own dataset.
    * Choices: ["unlearncanvas", "i2p", "generic"]
    * Example: "unlearncanvas"

* **template**: Type of template to use during training.
    * Choices: ["object", "style", "i2p"]
    * Example: "style"

* **template_name**: Name of the specific concept or style to be erased.
    * Choices: ["self-harm", "Abstractionism"]
    * Example: "Abstractionism"

**Output Configurations**

* **output_dir**: Directory where the fine-tuned models and results will be saved.
    * Type: str
    * Example: "outputs/erase_diff/finetuned_models"

**Sampling and Image Configurations**

* **use_sample**: Flag to indicate whether a sample dataset should be used for training.
    * Type: bool
    * Example: True

* **guided_concepts**: Concepts to guide the editing process.
    * Type: str
    * Example: "Nature, Abstract"

* **technique**: Specifies the editing technique.
    * Choices: ["replace", "tensor"]
    * Example: "replace"

* **preserve_scale**: Scale for preservation during the editing process.
    * Type: float
    * Example: 0.5

* **preserve_number**: Number of items to preserve during editing.
    * Type: int
    * Example: 10

* **erase_scale**: Scale for erasure during the editing process.
    * Type: float
    * Example: 0.8

* **lamb**: Lambda parameter for controlling balance during editing.
    * Type: float
    * Example: 0.01

* **add_prompts**: Flag to indicate whether additional prompts should be used.
    * Type: bool
    * Example: True

**Device Configuration**

* **devices**: Specifies the CUDA devices to be used for training (comma-separated).
    * Type: str (Comma-separated)
    * Example: "0,1"



