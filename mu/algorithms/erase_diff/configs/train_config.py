import os
from mu.core.base_config import BaseConfig


class EraseDiffConfig(BaseConfig):

    def __init__(self, **kwargs):
        self.train_method = "xattn"
        self.alpha = 0.1
        self.epochs = 1
        self.K_steps = 2
        self.lr = 5e-5
        self.model_config_path = "mu/algorithms/erase_diff/configs/model_config.yaml"
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"
        self.raw_dataset_dir = "data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/erase_diff/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"
        self.output_dir = "outputs/erase_diff/finetuned_models"
        self.separator = None
        self.image_size = 512
        self.interpolation = "bicubic"
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.devices = "0"
        self.use_sample = True
        self.num_workers = 4
        self.pin_memory = True

        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        TRAIN_METHODS = [
            "noxattn",
            "selfattn",
            "xattn",
            "full",
            "notime",
            "xlayer",
            "selflayer",
        ]
        if self.epochs <= 0:
            raise ValueError("epochs should be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if self.image_size <= 0:
            raise ValueError("Image size should be positive.")
        if self.train_method not in TRAIN_METHODS:
            raise ValueError(f"Invalid train method. Choose from {TRAIN_METHODS}")
        if self.dataset_type not in ["unlearncanvas", "i2p"]:
            raise ValueError(
                "Invalid dataset type. Choose from ['unlearncanvas', 'i2p']"
            )

        # check if folder exists
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            raise FileNotFoundError(
                f"Directory {self.processed_dataset_dir} does not exist."
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model config file {self.model_config_path} does not exist."
            )
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")


erase_diff_train_config_quick_canvas = EraseDiffConfig()
erase_diff_train_config_quick_canvas.dataset_type = "unlearncanvas"
erase_diff_train_config_quick_canvas.raw_dataset_dir = (
    "data/quick-canvas-dataset/sample"
)

erase_diff_train_config_i2p = EraseDiffConfig()
erase_diff_train_config_i2p.dataset_type = "i2p"
erase_diff_train_config_i2p.raw_dataset_dir = "data/i2p-dataset/sample"
