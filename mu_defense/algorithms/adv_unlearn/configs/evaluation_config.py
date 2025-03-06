# mu_defense/algorithms/adv_unlearn/configs/evaluation_config.py

import os

from mu.core.base_config import BaseConfig

class MUDefenseEvaluationConfig(BaseConfig):
    def __init__(self):
        self.model_name = "SD-v1-4"
        self.target_ckpt = ""
        self.save_path = "output/images"
        self.prompts_path = "data/prompts/visualization_example.csv"
        self.guidance_scale = 7.5
        self.image_size = 512
        self.ddim_steps = 100
        self.num_samples = 1
        self.from_case = 0
        self.folder_suffix = ""
        self.origin_or_target = "target" #target or origin
        self.encoder_model_name_or_path = "CompVis/stable-diffusion-v1-4"

        self.gen_imgs_path = "outputs/adv_unlearn/models_visualizations_imagenette/SD-v1-4/"
        self.coco_imgs_path = "coco_dataset/extracted_files/coco_sample"
        self.prompt_path = "data/prompts/coco_10k.csv"
        self.classify_prompt_path = "data/prompts/imagenette_5k.csv"
        self.devices = "0,0"
        self.classification_model_path = "openai/clip-vit-base-patch32"
        self.output_path = "outputs/adv_unlearn/evaluation"

    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt dataset file {self.prompt_path} does not exist.")
        
mu_defense_evaluation_config = MUDefenseEvaluationConfig()