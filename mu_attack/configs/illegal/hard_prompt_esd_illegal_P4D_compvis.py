# mu_attack/configs/illegal/hard_prompt_esd_illegal_P4D_compvis.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class OverallConfig(BaseConfig):
    def __init__(self,
                 task="P4D",
                 attacker="hard_prompt",
                 logger="json",
                 resume=None):
        self.task = task
        self.attacker = attacker
        self.logger = logger
        self.resume = resume


class TaskConfig(BaseConfig):
    def __init__(self,
                 concept="harm",
                 compvis_ckpt_path="outputs/scissorhands/finetuned_models/scissorhands_Abstractionism_model.pth",
                 compvis_config_path="mu/algorithms/scissorhands/configs/model_config.yaml", #TODO fix this path
                 cache_path=".cache",
                 dataset_path="files/dataset/illegal",
                 criterion="l2",
                 classifier_dir=None,
                 backend="compvis"):
        self.concept = concept
        self.compvis_ckpt_path = compvis_ckpt_path
        self.compvis_config_path = compvis_config_path
        self.cache_path = cache_path
        self.dataset_path = dataset_path
        self.criterion = criterion
        self.classifier_dir = classifier_dir
        self.backend = backend


class AttackerConfig(BaseConfig):
    def __init__(self,
                 insertion_location="prefix_k",
                 k=5,
                 iteration=40,
                 seed_iteration=1,
                 attack_idx=0,
                 eval_seed=0,
                 universal=False,
                 sequential=True,
                 hard_prompt=None):
        self.insertion_location = insertion_location
        self.k = k
        self.iteration = iteration
        self.seed_iteration = seed_iteration
        self.attack_idx = attack_idx
        self.eval_seed = eval_seed
        self.universal = universal
        self.sequential = sequential
        self.hard_prompt = HardPromptConfig(**hard_prompt) if hard_prompt else HardPromptConfig()


class HardPromptConfig(BaseConfig):
    def __init__(self, lr=0.01, weight_decay=0.1):
        self.lr = lr
        self.weight_decay = weight_decay


class LoggerConfig(BaseConfig):
    def __init__(self,
                 json_config=None):
        self.json = JSONLoggerConfig(**json_config) if json_config else JSONLoggerConfig()


class JSONLoggerConfig(BaseConfig):
    def __init__(self, root="files/results/hard_prompt_esd_illegal_P4D_scissorhands"):
        self.root = root


class HardPromptIllegalConfigCompvis(BaseConfig):
    def __init__(self,
                 overall=None,
                 task=None,
                 attacker=None,
                 logger=None):
        super().__init__()
        self.overall = OverallConfig(**overall) if overall else OverallConfig()
        self.task = TaskConfig(**task) if task else TaskConfig()
        self.attacker = AttackerConfig(**attacker) if attacker else AttackerConfig()
        self.logger = LoggerConfig(**logger) if logger else LoggerConfig()

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if self.task:
            if not os.path.exists(self.task.compvis_ckpt_path):
                raise FileNotFoundError(f"Checkpoint path {self.task.compvis_ckpt_path} does not exist.")
            if not os.path.exists(self.task.compvis_config_path):
                raise FileNotFoundError(f"Config path {self.task.compvis_config_path} does not exist.")
            if not os.path.exists(self.task.dataset_path):
                raise FileNotFoundError(f"Dataset path {self.task.dataset_path} does not exist.")
        if self.logger and self.logger.json:
            if not os.path.exists(self.logger.json.root):
                raise FileNotFoundError(f"Logger root directory {self.logger.json.root} does not exist.")


hard_prompt_esd_illegal_P4D_compvis = HardPromptIllegalConfigCompvis()