import os
from pathlib import Path

from mu.core.base_config import BaseConfig


class DataHandlerSubConfig(BaseConfig):
    def __init__(self, target: str, size: int = 512):
        self.target = target
        self.params = {"size": size}


class DataHandlerParamsConfig(BaseConfig):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 1,
        wrap: bool = False,
        train: DataHandlerSubConfig = None,
        train2: DataHandlerSubConfig = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wrap = wrap
        self.train = train
        self.train2 = train2


class DataConfig(BaseConfig):
    def __init__(
        self,
        target: str = "mu.algorithms.concept_ablation.data_handler.ConceptAblationDataHandler",
        params: DataHandlerParamsConfig = None,
    ):
        self.target = target
        self.params = params


class ImageLoggerCallbackParams(BaseConfig):
    def __init__(
        self,
        batch_frequency: int = 20000,
        save_freq: int = 10000,
        max_images: int = 8,
        increase_log_steps: bool = False,
    ):
        self.batch_frequency = batch_frequency
        self.save_freq = save_freq
        self.max_images = max_images
        self.increase_log_steps = increase_log_steps


class ImageLoggerCallbackConfig(BaseConfig):
    def __init__(
        self,
        target: str = "mu.algorithms.concept_ablation.callbacks.ImageLogger",
        params: ImageLoggerCallbackParams = None,
    ):
        self.target = target
        self.params = params


class ModelCheckpointParamsConfig(BaseConfig):
    def __init__(self, every_n_train_steps: int = 10000):
        self.every_n_train_steps = every_n_train_steps


class ModelCheckpointConfig(BaseConfig):
    def __init__(self, params: ModelCheckpointParamsConfig = None):
        self.params = params


class TrainerConfig(BaseConfig):
    def __init__(self, max_steps: int = 2000):
        self.max_steps = max_steps


class CallbacksConfig(BaseConfig):
    def __init__(self, image_logger: ImageLoggerCallbackConfig = None):
        self.image_logger = image_logger


class LightningConfig(BaseConfig):
    def __init__(
        self,
        callbacks: CallbacksConfig = None,
        modelcheckpoint: ModelCheckpointConfig = None,
        trainer: TrainerConfig = None,
    ):
        self.callbacks = callbacks
        self.modelcheckpoint = modelcheckpoint
        self.trainer = trainer


class ConceptAblationConfig(BaseConfig):
    def __init__(self, **kwargs):
        current_dir = Path(__file__).parent
        self.seed = 23
        self.scale_lr = True
        self.caption_target = "Abstractionism Style"
        self.regularization = True
        self.n_samples = 1
        self.train_size = 2
        self.base_lr = 2.0e-06
        self.config_path = current_dir / "train_config.yaml"
        self.model_config_path = current_dir / "model_config.yaml"
        self.ckpt_path = "models/compvis/style50/compvis.ckpt"
        self.raw_dataset_dir = "/home/ubuntu/Projects/balaram/msu_unlearningalgorithm/data/quick-canvas-dataset/sample"
        self.processed_dataset_dir = "mu/algorithms/concept_ablation/data"
        self.dataset_type = "unlearncanvas"
        self.template = "style"
        self.template_name = "Abstractionism"
        self.lr = 5e-5
        self.output_dir = "outputs/concept_ablation/finetuned_models"
        self.devices = "0,"
        self.reg_caption = ""
        self.reg_caption2 = ""
        self.datapath = ""
        self.reg_datapath = None
        self.datapath2 = ""
        self.reg_datapath2 = None
        self.caption2 = ""
        self.reg_caption2 = ""
        self.loss_type_reverse = "model-based"
        self.use_sample = True
        self.data = DataConfig(
            target="mu.algorithms.concept_ablation.data_handler.ConceptAblationDataHandler",
            params=DataHandlerParamsConfig(
                batch_size=1,
                num_workers=1,
                wrap=False,
                train=DataHandlerSubConfig(
                    target="mu.algorithms.concept_ablation.src.finetune_data.MaskBase",
                    size=512,
                ),
                train2=DataHandlerSubConfig(
                    target="mu.algorithms.concept_ablation.src.finetune_data.MaskBase",
                    size=512,
                ),
            ),
        )
        self.lightning = LightningConfig(
            callbacks=CallbacksConfig(
                image_logger=ImageLoggerCallbackConfig(
                    target="mu.algorithms.concept_ablation.callbacks.ImageLogger",
                    params=ImageLoggerCallbackParams(
                        batch_frequency=20000,
                        save_freq=10000,
                        max_images=8,
                        increase_log_steps=False,
                    ),
                )
            ),
            modelcheckpoint=ModelCheckpointConfig(
                params=ModelCheckpointParamsConfig(every_n_train_steps=2)
            ),
            trainer=TrainerConfig(max_steps=200),
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_config(self):
        if not os.path.exists(self.raw_dataset_dir):
            raise FileNotFoundError(f"Directory {self.raw_dataset_dir} does not exist.")
        if not os.path.exists(self.processed_dataset_dir):
            os.makedirs(self.processed_dataset_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {self.ckpt_path} does not exist.")
        if self.lr <= 0:
            raise ValueError("Learning rate (lr) should be positive.")
        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(
                f"Model config file {self.model_config_path} does not exist."
            )


concept_ablation_train_mu = ConceptAblationConfig(
    dataset_type="unlearncanvas", raw_dataset_dir="data/quick-canvas-dataset/sample"
)

concept_ablation_train_i2p = ConceptAblationConfig(
    dataset_type="i2p", raw_dataset_dir="data/i2p-dataset/sample"
)
