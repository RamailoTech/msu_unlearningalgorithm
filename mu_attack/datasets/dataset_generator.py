# mu_attack/datasets/dataset_generator.py

import torch
import os

import pandas as pd
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm

class DatasetGenerator:
    def __init__(self, device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=100, num_samples=10, cache_dir='./ldm_pretrained'):
        self.device = device
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.num_samples = num_samples
        self.cache_dir = cache_dir

        self._load_models()

    def _load_models(self):
        model_name = "CompVis/stable-diffusion-v1-4"
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", cache_dir=self.cache_dir).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", cache_dir=self.cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", cache_dir=self.cache_dir).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", cache_dir=self.cache_dir).to(self.device)
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    def generate_images(self, prompts_path, save_path, concept='default', from_case=0, ckpt=None):
        if ckpt:
            self.unet.load_state_dict(torch.load(ckpt, map_location=self.device))

        df = pd.read_csv(prompts_path)
        folder_path = os.path.join(save_path, concept)
        os.makedirs(f'{folder_path}/imgs', exist_ok=True)
        repeated_rows = []

        for i, row in df.iterrows():
            prompt = [str(row.prompt)] * self.num_samples
            seed = row.get('evaluation_seed', row.get('sd_seed', 42))
            case_number = row.get('case_number', i)
            
            repeated_rows.extend([row]*self.num_samples)

            if case_number < from_case:
                continue

            height = row.get('sd_image_height', self.image_size)
            width = row.get('sd_image_width', self.image_size)
            guidance_scale = row.get('sd_guidance_scale', self.guidance_scale)

            generator = torch.manual_seed(seed)
            batch_size = len(prompt)

            text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            max_length = text_input.input_ids.shape[-1]

            uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn((batch_size, self.unet.config.in_channels, height // 8, width // 8), generator=generator).to(self.device)
            self.scheduler.set_timesteps(self.ddim_steps)
            latents = latents * self.scheduler.init_noise_sigma

            self.scheduler.set_timesteps(self.ddim_steps)

            for t in tqdm(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = self.vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")

            pil_images = [Image.fromarray(img) for img in image]
            for num, im in enumerate(pil_images):
                im.save(f"{folder_path}/imgs/{case_number}_{num}.png")
        new_df = pd.DataFrame(repeated_rows)
        new_df.to_csv(os.path.join(folder_path, 'prompts.csv'), index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate datasets using Diffusers in a class-based approach.')
    parser.add_argument('--prompts_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--concept', type=str, default='default')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--from_case', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default='./ldm_pretrained')
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()

    generator = DatasetGenerator(
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir
    )

    generator.generate_images(
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        concept=args.concept,
        from_case=args.from_case,
        ckpt=args.ckpt
    )
