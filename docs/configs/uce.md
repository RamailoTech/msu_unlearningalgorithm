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
