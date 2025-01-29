# mu_attack/core/base_config.py

import os
from pydantic import BaseModel
from typing import Optional, Literal

class AttackerConfig(BaseModel):
    insertion_location: str = "prefix_k"
    k: int = 5
    iteration: int = 40
    seed_iteration: int = 1
    attack_idx: int = 0
    eval_seed: int = 0
    universal: bool = False
    sequential: bool = False
    lr : Optional[float] = None
    weight_decay : Optional[float] = None
    hard_prompt: Optional[dict] = {}

class OverallConfig(BaseModel):
    task: Literal["P4D","classifier"]
    attacker: Literal["hard_prompt", "no_attack", "random", "seed_search", "text_grad"]
    logger: str = "json"
    resume: Optional[str] = None

class LoggerConfig(BaseModel):
    json: dict

class TaskConfig(BaseModel):
    concept: str = "nudity"
    compvis_ckpt_path: str = None
    compvis_config_path: str = None
    cache_path: str = ".cache" 
    dataset_path: str = None
    criterion: str = "l2"
    backend: Literal["compvis", "diffusers"]
    sampling_step_num : Optional[int]
    sld: Optional[str] 
    sld_concept: Optional[str]
    negative_prompt: Optional[str]
    classifier_dir: Optional[str] 
    

class BaseConfig(BaseModel):
    overall: OverallConfig
    task: TaskConfig
    attacker: AttackerConfig
    logger: LoggerConfig
    

    def validate_config(self):
        """
        Perform validation to ensure required paths exist.
        """
        pass


