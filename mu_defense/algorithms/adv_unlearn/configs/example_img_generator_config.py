# mu_defense/algorithms/adv_unlearn/configs/example_img_generator_config.py

import os

from mu.core.base_config import BaseConfig

class ImageGeneratorConfig(BaseConfig):
    def __init__(self):
        self.model_name = "SD-v1-4"
        self.target_ckpt = ""
        self.save_path = ""
        self.prompts_path = "data/prompts/visualization_example.csv"
        self.device = "0"
        self.guidance_scale = 7.5
        self.image_size = 512
        self.ddim_steps = 100
        self.num_samples = 1
        self.from_case = 0
        self.folder_suffix = ""
        self.origin_or_target = "target" #target or origin
        self.encoder_model_name_or_path = "CompVis/stable-diffusion-v1-4"


    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt dataset file {self.prompt_path} does not exist.")
        
example_image_generator_config = ImageGeneratorConfig()