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
    no_attack: Optional[dict] = {}
    text_grad: Optional[dict] = {}

class OverallConfig(BaseModel):
    task: Literal["P4D","classifier"]
    attacker: Literal["hard_prompt", "no_attack", "random", "seed_search", "text_grad"]
    logger: str = "json"
    resume: Optional[str] = None

class LoggerConfig(BaseModel):
    json: dict

class TaskConfig(BaseModel):
    concept: str = "nudity"
    cache_path: str = ".cache" 
    dataset_path: str = None
    criterion: str = "l2"
    backend: Literal["compvis", "diffusers"]
    converted_model_folder_path: str = "outputs"
    model_name : Optional[str] = "SD-v1-4"
    save_diffuser: Optional[bool]  = True
    compvis_ckpt_path: Optional[str] = None
    compvis_config_path: Optional[str] = None
    diffusers_model_name_or_path : Optional[str] = None
    target_ckpt : Optional[str] = None
    sampling_step_num : Optional[int] = None
    sld: Optional[str] = None
    sld_concept: Optional[str] = None
    negative_prompt: Optional[str] = None
    classifier_dir: Optional[str] =  None
    diffusers_config_file: Optional[str] = None
    
    

class BaseConfig(BaseModel):
    overall: OverallConfig
    task: TaskConfig
    attacker: AttackerConfig
    logger: LoggerConfig
    
