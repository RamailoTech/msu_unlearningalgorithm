#mu/algorithms/esd/sampler.py

import os
import logging
from typing import Optional
import numpy as np

import torch
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything

from stable_diffusion.constants.const import theme_available, class_available
from mu.core import BaseSampler
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from mu.algorithms.esd.algorithm import ESDModel
from mu.helpers import sample_model, load_config
from mu.helpers.utils import load_ckpt_from_config


#TODO remove this
theme_available = ['Abstractionism', 'Bricks', 'Cartoon']
class_available = ['Architectures', 'Bears', 'Birds']


class ESDSampler(BaseSampler):
    """Sampler for the ESD algorithm."""

    def __init__(self, model: ESDModel, config: dict, device):
        self.model, self.model_orig = model.models
        self.config = config
        self.device = device
        self.ddim_steps = self.config['ddim_steps']
        self.ddim_eta = 0
        self.samplers = self.load_samplers(self.model, self.model_orig)

    def load_samplers(self, model, model_orig):
        """
        Load the samplers
        """
        sampler = DDIMSampler(model)
        sampler_orig = DDIMSampler(model_orig)
        return (sampler, sampler_orig)
            
    def sample(self, c, h, w, scale, start_code=None, num_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
        '''Generates samples using the model and sampler.

        Parameters:
            c (torch.Tensor): The conditioning input
            h (int): Height of the output image
            w (int): Width of the output image
            scale (float): The unconditional guidance scale
            start_code (torch.Tensor, optional): Starting noise tensor. Defaults to None.
            num_samples (int, optional): Number of samples to generate. Defaults to 1.
            t_start (int, optional): Starting timestep. Defaults to -1.
            log_every_t (int, optional): Log progress every t steps. Defaults to None.
            till_T (int, optional): Run sampling until timestep T. Defaults to None.
            verbose (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            torch.Tensor: Generated samples of shape (num_samples, channels, height, width)
        '''

        samples = sample_model(self.model, self.samplers[0], c, h, w, self.ddim_steps, scale, self.ddim_eta,
                               start_code=start_code, num_samples=num_samples, t_start=t_start,
                               log_every_t=log_every_t, till_T=till_T, verbose=verbose)
        return samples


class ESDEvaluatorSampler(BaseSampler):
    """ESD Image Generator class extending a hypothetical BaseImageGenerator."""

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the ESDSampler with a YAML config (or dict).
        
        Args:
            config (Dict[str, Any]): Dictionary of hyperparams / settings.
            **kwargs: Additional keyword arguments that can override config entries.
        """
        super().__init__()

        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sampler = None
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> None:
        """
        Load the model using `config` and initialize the sampler.
        """
        self.logger.info("Loading model...")
        model_ckpt_path = self.config["ckpt_path"]
        model_config = load_config(self.config["model_config"])
        self.model = load_ckpt_from_config(model_config, model_ckpt_path, verbose=True)
        self.model.to(self.device)
        self.model.eval()
        self.sampler = DDIMSampler(self.model)
        self.logger.info("Model loaded and sampler initialized successfully.")

    def sample(self) -> None:
        """
        Sample (generate) images using the loaded model and sampler, based on the config.
        """
        steps = self.config["ddim_steps"]       
        cfg_text = self.config["cfg_text"]    
        seed = self.config["seed"]
        H = self.config["image_height"]
        W = self.config["image_width"]
        ddim_eta = self.config["ddim_eta"]
        output_dir = self.config["sampler_output_dir"]

        # Generate directories for each theme
        for test_theme in theme_available:
            theme_path = os.path.join(output_dir, test_theme)
            os.makedirs(theme_path, exist_ok=True)
        
        self.logger.info(f"Generating images and saving to {output_dir}")

        # Set random seed
        seed_everything(seed)
    
        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_',' ')} style."
                self.logger.info(f"Sampling prompt: {prompt}")
                
                with torch.no_grad():
                    with autocast(self.device):
                        with self.model.ema_scope():
                            # Prepare conditioning
                            uc = self.model.get_learned_conditioning([""])  
                            c  = self.model.get_learned_conditioning(prompt)
                            shape = [4, H // 8, W // 8] # downsampling factor 8

                            # Generate samples using the sampler
                            samples_ddim, _ = self.sampler.sample(
                                S=steps,
                                conditioning=c,
                                batch_size=1,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=cfg_text,
                                unconditional_conditioning=uc,
                                eta=ddim_eta,
                                x_T=None
                            )

                            # Convert generated samples to image
                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            assert len(x_samples_ddim) == 1

                            # Convert to uint8 format
                            x_sample = x_samples_ddim[0]
                            if isinstance(x_sample, torch.Tensor):
                                x_sample = (255. * x_sample.cpu().detach().numpy()).round()
                            else:
                                x_sample = (255. * x_sample).round()
                            x_sample = x_sample.astype(np.uint8)
                            img = Image.fromarray(x_sample)

                            # Save the generated image in the theme's directory
                            filename = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                            outpath = os.path.join(output_dir, test_theme, filename)
                            img.save(outpath)  # Saving image directly
                            self.logger.info(f"Image saved: {outpath}")

        self.logger.info("Image generation completed successfully.")

    def save_image(self, image: Image.Image, file_path: str) -> None:
        """
        Save an image to the specified path.
        """
        image.save(file_path)
        self.logger.info(f"Image saved at: {file_path}")