# mu_defense/algorithms/adv_unlearn/generate_example_image.py

import os
import pandas as pd
from tqdm.auto import tqdm

import torch
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler

from evaluation.core import BaseImageGenerator
from mu_defense.algorithms.adv_unlearn.configs import ImageGeneratorConfig

class ImageGenerator(BaseImageGenerator):
    """
    DiffusersImageGenerator generates images using Stable Diffusion models.
    
    The generator is configured via a config dict that includes:
        - model_name: str, e.g. 'SD-v1-4', 'SD-V2', etc.
        - save_path: str, folder to save images
        - prompts_path: str, CSV file with prompts and seeds
        - device: str (default 'cuda:0')
        - guidance_scale: float (default 7.5)
        - image_size: int (default 512)
        - ddim_steps: int (default 100)
        - num_samples: int (default 1)
        - from_case: int (default 0)
        - folder_suffix: str (default 'imagenette')
        - origin_or_target: str (default 'target')
    """
    
    def __init__(self, config: ImageGeneratorConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Extract configuration parameters
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        self.model_name = self.config.get("model_name")
        self.target_ckpt = self.config.get("target_ckpt")  # may be None
        self.save_path = self.config["save_path"]
        self.prompts_path = self.config["prompts_path"]
        self.guidance_scale = self.config.get("guidance_scale", 7.5)
        self.image_size = self.config.get("image_size", 512)
        self.ddim_steps = self.config.get("ddim_steps", 100)
        self.num_samples = self.config.get("num_samples", 1)
        self.from_case = self.config.get("from_case", 0)
        self.folder_suffix = self.config.get("folder_suffix", "imagenette")
        self.origin_or_target = self.config.get("origin_or_target", "target")

        # Parse device(s)
        self.device = [f"cuda:{int(d.strip())}" for d in self.config.get("device", "0").split(",")]

        # Determine default model directory based on model_name (used when no target checkpoint is provided)
        if self.model_name == 'SD-v1-4':
            self.dir_ = "CompVis/stable-diffusion-v1-4"
        elif self.model_name == 'SD-V2':
            self.dir_ = "stabilityai/stable-diffusion-2-base"
        elif self.model_name == 'SD-V2-1':
            self.dir_ = "stabilityai/stable-diffusion-2-1-base"
        else:
            self.dir_ = "CompVis/stable-diffusion-v1-4"
        
        # Set up final save path for generated images
        if self.origin_or_target == 'target':
            self.final_save_path = f'{self.save_path}_visualizations_{self.folder_suffix}'
        else:
            self.target_ckpt = None
            self.final_save_path = os.path.join(self.save_path, 'original_SD', f'visualizations_{self.folder_suffix}')
        os.makedirs(self.final_save_path, exist_ok=True)

        if self.target_ckpt is not None:
            # If the target checkpoint does NOT end with ".pt", assume it's a Diffusers-format checkpoint.
            if not self.target_ckpt.endswith(".pt"):
                print("Loading Diffusers checkpoint from directory:", self.target_ckpt)
                self.vae = AutoencoderKL.from_pretrained(os.path.join(self.target_ckpt, "vae")).to(self.device[0])
                self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.target_ckpt, "tokenizer"))
                self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(self.target_ckpt, "text_encoder")).to(self.device[0])
                self.unet = UNet2DConditionModel.from_pretrained(os.path.join(self.target_ckpt, "unet")).to(self.device[0])
            else:
                print("Loading CompVis checkpoint from file:", self.target_ckpt)
                # Load default components from hub first
                self.vae = AutoencoderKL.from_pretrained(self.dir_, subfolder="vae").to(self.device[0])
                self.tokenizer = CLIPTokenizer.from_pretrained(self.dir_, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(self.dir_, subfolder="text_encoder").to(self.device[0])
                self.unet = UNet2DConditionModel.from_pretrained(self.dir_, subfolder="unet").to(self.device[0])
                # Override with target checkpoint weights (for UNet or text encoder)
                if 'TextEncoder' not in self.target_ckpt:
                    self.unet.load_state_dict(torch.load(self.target_ckpt))
                else:
                    state_dict = self.extract_text_encoder_ckpt(self.target_ckpt)
                    self.text_encoder.load_state_dict(state_dict, strict=False)
        # Initialize the scheduler (using LMSDiscreteScheduler; adjust parameters as needed)
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )


    @staticmethod
    def get_openai_diffuser_transformer(diffuser_ckpt: dict) -> dict:
        """
        Transform a diffuser checkpoint to a different format.
        """
        open_ckpt = {}
        for i in range(12):
            open_ckpt[f'transformer.resblocks.{i}.ln_1.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm1.weight']
            open_ckpt[f'transformer.resblocks.{i}.ln_1.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm1.bias']

            q_proj_weight = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.q_proj.weight']
            k_proj_weight = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.k_proj.weight']
            v_proj_weight = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.v_proj.weight']
            proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
            q_proj_bias = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.q_proj.bias']
            k_proj_bias = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.k_proj.bias']
            v_proj_bias = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.v_proj.bias']
            proj_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
            open_ckpt[f'transformer.resblocks.{i}.attn.in_proj_weight'] = proj_weight
            open_ckpt[f'transformer.resblocks.{i}.attn.in_proj_bias'] = proj_bias

            open_ckpt[f'transformer.resblocks.{i}.attn.out_proj.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.out_proj.weight']
            open_ckpt[f'transformer.resblocks.{i}.attn.out_proj.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.self_attn.out_proj.bias']

            open_ckpt[f'transformer.resblocks.{i}.ln_2.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm2.weight']
            open_ckpt[f'transformer.resblocks.{i}.ln_2.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.layer_norm2.bias']
            open_ckpt[f'transformer.resblocks.{i}.mlp.c_fc.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc1.weight']
            open_ckpt[f'transformer.resblocks.{i}.mlp.c_fc.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc1.bias']
            open_ckpt[f'transformer.resblocks.{i}.mlp.c_proj.weight'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc2.weight']
            open_ckpt[f'transformer.resblocks.{i}.mlp.c_proj.bias'] = diffuser_ckpt[f'text_model.encoder.layers.{i}.mlp.fc2.bias']

        open_ckpt['ln_final.weight'] = diffuser_ckpt['text_model.final_layer_norm.weight']
        open_ckpt['ln_final.bias'] = diffuser_ckpt['text_model.final_layer_norm.bias']
        return open_ckpt

    @staticmethod
    def extract_text_encoder_ckpt(ckpt_path: str) -> dict:
        """
        Extract only the text encoder weights from a checkpoint.
        """
        full_ckpt = torch.load(ckpt_path)
        new_ckpt = {}
        for key in full_ckpt.keys():
            if 'text_encoder.text_model' in key:
                new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
        return new_ckpt

    def generate_images(self):
        """
        Generate images from a CSV file containing prompts, seeds, and case numbers.
        
        Expected CSV columns:
            - case_number
            - prompt
            - evaluation_seed
        """
        df = pd.read_csv(self.prompts_path)
        folder_path = os.path.join(self.final_save_path, self.model_name)
        os.makedirs(folder_path, exist_ok=True)
        
        for _, row in df.iterrows():
            prompt = [str(row.prompt)] * self.num_samples
            seed = row.evaluation_seed
            case_number = row.case_number
            if case_number < self.from_case:
                continue

            height = self.image_size
            width = self.image_size
            num_inference_steps = self.ddim_steps

            generator = torch.manual_seed(seed)
            batch_size = len(prompt)

            # Tokenize and encode the prompt
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device[0]))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device[0]))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Prepare latent noise and scale by scheduler noise sigma
            latents = torch.randn(
                (batch_size, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
            ).to(self.device[0])
            self.scheduler.set_timesteps(num_inference_steps)
            latents = latents * self.scheduler.init_noise_sigma

            # Denoising loop
            for t in tqdm(self.scheduler.timesteps, desc=f"Generating case {case_number}"):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Decode the latents to images
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(img) for img in images]

            # Save images
            for num, im in enumerate(pil_images):
                im.save(os.path.join(folder_path, f"{case_number}_{num}.png"))
