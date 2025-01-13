#mu/algorithms/unified_concept_editing/sampler.py

import os
import logging

from PIL import Image
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler

from mu.core.base_sampler import BaseSampler        
from stable_diffusion.constants.const import theme_available, class_available

#TODO remove this
theme_available = ['Abstractionism', 'Bricks', 'Cartoon']
class_available = ['Architectures', 'Bears', 'Birds']

class UnifiedConceptEditingSampler(BaseSampler):
    """Unified Concept editing Image Generator class extending a hypothetical BaseImageGenerator."""

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the UnifiedConceptEditingSampler with a YAML config (or dict).
        
        Args:
            config (Dict[str, Any]): Dictionary of hyperparams / settings.
            **kwargs: Additional keyword arguments that can override config entries.
        """
        super().__init__()

        self.config = config
        self.device = self.config['devices'][0]
        self.model = None
        self.sampler = None
        self.tokenizer = None
        self.text_encoder = None
        self.scheduler = None
        self.unet = None
        self.vae = None
        self.unet = None
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> None:
        """
        Load the model using `config` and initialize the sampler.
        """
        self.logger.info("Loading model...")
        model_ckpt_path = self.config["ckpt_path"]
        pipeline_path = self.config["pipeline_path"]
        seed = self.config["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(pipeline_path, subfolder="vae", cache_dir="./cache", torch_dtype=torch.float16)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(pipeline_path, subfolder="tokenizer", cache_dir="./cache", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(pipeline_path, subfolder="text_encoder", cache_dir="./cache", torch_dtype=torch.float16)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(pipeline_path, subfolder="unet", cache_dir="./cache", torch_dtype=torch.float16)
        self.unet.load_state_dict(torch.load(model_ckpt_path, map_location=self.device))
        self.unet.to(torch.float16)
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000)

        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)

        self.logger.info("Model loaded and sampler initialized successfully.")

    def sample(self) -> None:
        """
        Sample (generate) images using the loaded model and sampler, based on the config.
        """
        steps = self.config["ddim_steps"]
        theme = self.config["theme"]  
        batch_size = self.config["batch_size"]       
        cfg_text = self.config["cfg_text"]    
        seed = self.config["seed"]
        height = self.config["image_height"]
        width = self.config["image_width"]
        ddim_eta = self.config["ddim_eta"]
        output_dir = self.config["sampler_output_dir"]

        for test_theme in theme_available:
            theme_path = os.path.join(output_dir, test_theme)
            os.makedirs(theme_path, exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")
       
        for test_theme in theme_available:
            for object_class in class_available:
                filename = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                output_path = os.path.join(output_dir, test_theme, filename)
                if os.path.exists(output_path):
                    print(f"Detected! Skipping {output_path}")
                    continue
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                generator = torch.manual_seed(seed)  # Seed generator to create the inital latent noise
                text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,
                                    return_tensors="pt")
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                latents = torch.randn(
                    (batch_size, self.unet.in_channels, height // 8, width // 8),
                    generator=generator,
                )
                latents = latents.to(self.device)

                self.scheduler.set_timesteps(steps)

                latents = latents * self.scheduler.init_noise_sigma

                from tqdm.auto import tqdm
                self.scheduler.set_timesteps(steps)
                # the model is trained in fp16, use mixed precision forward pass
                with torch.cuda.amp.autocast():
                    # predict the noise residual
                    with torch.no_grad():
                        for t in tqdm(self.scheduler.timesteps):
                            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                            latent_model_input = torch.cat([latents] * 2)

                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                            # perform guidance
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + cfg_text * (noise_pred_text - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # the model is trained in fp16, use mixed precision forward pass
                with torch.cuda.amp.autocast():
                    # scale and decode the image latents with vae
                    latents = 1 / 0.18215 * latents
                    with torch.no_grad():
                        image = self.vae.decode(latents).sample

                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                    images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images][0]
                self.save_image(pil_images, output_path)

        self.logger.info("Image generation completed.")

    def save_image(self, image: Image.Image, file_path: str) -> None:
        """
        Save an image to the specified path.
        """
        image.save(file_path)
        self.logger.info(f"Image saved at: {file_path}")


