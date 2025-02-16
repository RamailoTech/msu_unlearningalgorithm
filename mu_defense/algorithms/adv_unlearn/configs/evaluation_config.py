# mu_defense/algorithms/adv_unlearn/configs/evaluation_config.py

import os

from mu.core.base_config import BaseConfig

class MUDefenseEvaluationConfig(BaseConfig):
    def __init__(self):
        self.job = "fid"
        self.gen_imgs_path = ""
        self.coco_imgs_path = ""
        self.prompt_path = "data/prompts/coco_10k.csv"
        self.classify_prompt_path = "data/prompts/imagenette_5k.csv"
        self.devices = "0,0"
        self.classification_model_path = "openai/clip-vit-base-patch32"

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt dataset file {self.prompt_path} does not exist.")
        
mu_defense_evaluation_config = MUDefenseEvaluationConfig()