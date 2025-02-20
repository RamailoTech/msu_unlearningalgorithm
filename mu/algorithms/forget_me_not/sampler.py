# mu/algorithms/forget_me_not/sampler.py

import os
import torch
import logging

from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline

from mu.core.base_sampler import BaseSampler  
from mu.datasets.constants import *

class ForgetMeNotSampler(BaseSampler):
    """ForgetMeNot Image Generator class extending a hypothetical BaseImageGenerator."""

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the ForgetMeNotSampler with a YAML config (or dict).
        
        Args:
            config (Dict[str, Any]): Dictionary of hyperparams / settings.
            **kwargs: Additional keyword arguments that can override config entries.
        """
        super().__init__()

        self.config = config
        self.device = self.config['devices'][0]
        self.pipe = None
        self.sampler = None
        self.use_sample = self.config.get('use_sample')
        self.theme_available = uc_sample_theme_available_eval if self.use_sample else uc_theme_available
        self.class_available = uc_sample_class_available_eval if self.use_sample else uc_class_available
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> None:
        """
        Load the model using `config` and initialize the sampler.
        """
        self.logger.info("Loading model...")
        
        model_ckpt_path = f"{self.config['ckpt_path']}"

        seed = self.config['seed']

         # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.pipe = StableDiffusionPipeline.from_pretrained(model_ckpt_path, torch_dtype=torch.float16).to(self.device)

        self.logger.info("Model loaded and sampler initialized successfully.")


    def dummy(self,images, **kwargs):
        return images, [False]

    def sample(self) -> None:
        steps = self.config["ddim_steps"]
        theme = self.config["forget_theme"]         
        cfg_text_list = self.config["cfg_text_list"]    
        seed = self.config["seed"]
        H = self.config["image_height"]
        W = self.config["image_width"]
        ddim_eta = self.config["ddim_eta"]
        output_dir = self.config["sampler_output_dir"]

        for test_theme in self.theme_available:
            theme_path = os.path.join(output_dir, test_theme)
            os.makedirs(theme_path, exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")

        # Disable NSFW checker
        self.pipe.safety_checker = self.dummy

        for test_theme in self.theme_available:
            for object_class in self.class_available:
                for cfg_text in cfg_text_list:
                    prompt = f"A {object_class} image in {test_theme} style"
                    self.logger.info(f"Generating: {prompt}")
                    
                    # Generate the image using the pipeline
                    generated_image = self.pipe(prompt=prompt, width=W, height=H, num_inference_steps=steps, guidance_scale=cfg_text).images[0]
                    filename = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                    output_path = os.path.join(output_dir, test_theme, filename)
                    self.save_image(generated_image, output_path)

        self.logger.info("Image generation completed.")

    def save_image(self, image: Image.Image, file_path: str) -> None:
        """
        Save an image to the specified path.
        """
        image.save(file_path)
        self.logger.info(f"Image saved at: {file_path}")