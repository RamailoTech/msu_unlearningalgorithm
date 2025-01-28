from abc import ABC, abstractmethod
from typing import Optional


class BaseConfig(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def validate_config(self):
        pass

    
    def to_dict(self):
        """
        Convert the entire configuration object to a dictionary.
        """
        pass


class OverallConfig(BaseConfig):
    def __init__(self,
                 task: str,
                 attacker: str,
                 logger: str,
                 resume: Optional[str] = None,
                 **kwargs):
        self.task = task
        self.attacker = attacker
        self.logger = logger
        self.resume = resume

        for key, value in kwargs.items():
            setattr(self, key, value)


class TaskConfigCompvis(BaseConfig):
    def __init__(self,
                 concept: str,
                 compvis_ckpt_path: str,
                 compvis_config_path: str,
                 cache_path: str,
                 dataset_path: str,
                 criterion: str,
                 sampling_step_num: int,
                 sld: str,
                 sld_concept: str,
                 negative_prompt: str,
                 backend: str,
                 **kwargs):
        self.concept = concept
        self.compvis_ckpt_path = compvis_ckpt_path
        self.compvis_config_path = compvis_config_path
        self.cache_path = cache_path
        self.dataset_path = dataset_path
        self.criterion = criterion
        self.sampling_step_num = sampling_step_num
        self.sld = sld
        self.sld_concept = sld_concept
        self.negative_prompt = negative_prompt
        self.backend = backend

        for key, value in kwargs.items():
            setattr(self, key, value)

class TaskConfigDiffuser(BaseConfig):
    def __init__(self,
                 concept: str,
                 diffusers_model_name_or_path: str,
                 target_ckpt: str,
                 cache_path: str,
                 dataset_path: str,
                 criterion: str,
                 sampling_step_num: int,
                 sld: str,
                 sld_concept: str,
                 negative_prompt: str,
                 backend: str,
                 **kwargs):
        self.concept = concept
        self.diffusers_model_name_or_path = diffusers_model_name_or_path
        self.target_ckpt = target_ckpt
        self.cache_path = cache_path
        self.dataset_path = dataset_path
        self.criterion = criterion
        self.sampling_step_num = sampling_step_num
        self.sld = sld
        self.sld_concept = sld_concept
        self.negative_prompt = negative_prompt
        self.backend = backend

        for key, value in kwargs.items():
            setattr(self, key, value)

class AttackerConfig(BaseConfig):
    def __init__(self,
                 insertion_location: str,
                 k: int,
                 iteration: int,
                 seed_iteration: int,
                 attack_idx: int,
                 eval_seed: int,
                 universal: bool,
                 sequential: bool,
                 **kwargs):
        self.insertion_location = insertion_location
        self.k = k
        self.iteration = iteration
        self.seed_iteration = seed_iteration
        self.attack_idx = attack_idx
        self.eval_seed = eval_seed
        self.universal = universal
        self.sequential = sequential

        for key, value in kwargs.items():
            setattr(self, key, value)

class NoAttackConfig(BaseConfig):
    """
    Class for specific 'no_attack' configuration details.
    """
    def __init__(self, dataset_path: str, **kwargs):
        self.dataset_path = dataset_path

        # Dynamically set additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

class NoAttackAttackerConfig(BaseConfig):
    """
    class for attacker for no attack
    """
    def __init__(self,
                 insertion_location: str,
                 k: int,
                 seed_iteration: int,
                 sequential: bool,
                 iteration: int,
                 attack_idx: int,
                 eval_seed: int,
                 universal: bool,
                 no_attack: dict,
                 **kwargs):
        self.insertion_location = insertion_location
        self.k = k
        self.seed_iteration = seed_iteration
        self.sequential = sequential
        self.iteration = iteration
        self.attack_idx = attack_idx
        self.eval_seed = eval_seed
        self.universal = universal
        self.no_attack = NoAttackConfig(**no_attack) if no_attack else None

        # Dynamically set additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)



class JSONLoggerConfig(BaseConfig):
    def __init__(self,
                 root: str,
                 name: str,
                 **kwargs):
        self.root = root
        self.name = name

        for key, value in kwargs.items():
            setattr(self, key, value)


class LoggerConfig(BaseConfig):
    def __init__(self,
                 json_config: Optional[dict] = None,
                 **kwargs):
        self.json = JSONLoggerConfig(**json_config) if json_config else JSONLoggerConfig(
            root=None, name=None)

        for key, value in kwargs.items():
            setattr(self, key, value)