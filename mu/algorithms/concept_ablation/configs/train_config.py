import os
from mu.core.base_config import BaseConfig


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
        self.model_config_path = "mu/algorithms/concept_ablation/configs/model_config.yaml"  # Path to model config
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
                "batch_size": 4,  # Batch size for training
                "num_workers": 4,  # Number of workers for loading data
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

        # Update properties based on provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        # Check if directories exist
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory {self.processed_dataset_dir} does not exist."
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Validate checkpoint path
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")

        # Validate device configuration
        devices = self.devices.split(",")
        for device in devices:
            if not device.isdigit():
                raise ValueError(
                    f"Invalid device {device}. Devices should be integers representing CUDA device IDs."
                )

        # Validate learning rate
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")

        # Validate model and data handler paths
        if not isinstance(self.data, dict):
            raise ValueError("Data configuration should be a dictionary.")
        if not isinstance(self.data["params"], dict):
            raise ValueError("Data parameters should be a dictionary.")

        # Validate Lightning configuration
        if not isinstance(self.lightning, dict):
            raise ValueError("Lightning configuration should be a dictionary.")
        if "callbacks" not in self.lightning or not isinstance(
            self.lightning["callbacks"], dict
        ):
            raise ValueError("Lightning callbacks should be a dictionary.")
        if "trainer" not in self.lightning or not isinstance(
            self.lightning["trainer"], dict
        ):
            raise ValueError("Lightning trainer should be a dictionary.")

        # Check if the model checkpoint exists
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model config file {self.model_config_path} does not exist."
            )


concept_ablation_train_config_quick_canvas = ConceptAblationConfig()
concept_ablation_train_config_quick_canvas.dataset_type = "unlearncanvas"
concept_ablation_train_config_quick_canvas.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

concept_ablation_train_config_i2p = ConceptAblationConfig()
concept_ablation_train_config_i2p.dataset_type = "i2p"
concept_ablation_train_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
