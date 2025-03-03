# mu_attack/tasks/sd_compvis.py

import sys
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer

from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_attack.tasks.utils.metrics.style_eval import init_classifier
from mu_attack.tasks.utils.metrics.object_eval import imagenet_ResNet50
from mu_attack.tasks.utils.metrics.harm_eval import q16_binary_classifier

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler

from mu_attack.core import BaseStableDiffusionPipeline

class BaseCompvisPipeline(BaseStableDiffusionPipeline):
    def __init__(self, config_path, ckpt_path, device, concept,  sld, sld_concept, negative_prompt, criterion, classifier_dir,cache_path,*args,**kwargs):
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.cache_path = cache_path
        self.device = device
        self.concept = concept 
        self.sld = sld 
        self.sld_concept = sld_concept 
        self.negative_prompt = negative_prompt
        self.criterion = torch.nn.L1Loss() if criterion == 'l1' else torch.nn.MSELoss()
        self.classifier_dir = classifier_dir

        # placeholders for loaded modules
        self.vae = None
        self.unet_sd = None
        self.target_unet_sd = None
        self.tokenizer = None
        self.text_encoder = None
        self.custom_text_encoder = None
        self.classifier = None
        self.scheduler = None
        self.processor = None 
        self.clip_model = None
        self.all_embeddings = None

        self.sampler = None

        super().__init__()


    def load_model(self):
        """
        Load the CompVis Stable Diffusion model from a .yaml config and .ckpt file,
        then set up any classifiers, text encoder references, etc.
        """
        if isinstance(self.config_path, (str, Path)):
            config = OmegaConf.load(self.config_path)
        else:
            config = self.config_path  # If already a config object

        pl_sd = torch.load(self.ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.model.cond_stage_model = None

        self.vae = self.model.first_stage_model
        self.unet_sd = self.model.model.diffusion_model

        self.target_unet_sd = deepcopy(self.unet_sd)

        self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer", cache_dir=self.cache_path)
        self.text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", cache_dir=self.cache_path).to(self.device)
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        if self.classifier_dir is not None:
            self.classifier = init_classifier(self.device, self.classifier_dir)
        elif self.concept in self.object_list:
            self.processor, self.classifier = imagenet_ResNet50(self.device)
        elif self.concept == 'harm':
            self.clip_model, self.classifier = q16_binary_classifier(self.device)

        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(
            ddim_num_steps=50,
            ddim_eta=0.0
        )

        # Freeze everything if needed
        for m in [self.vae, self.text_encoder, self.custom_text_encoder,self.unet_sd, self.target_unet_sd]:
            if m is not None:
                m.eval()
                m.requires_grad_(False)

    def str2id(self,prompt):
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt", truncation=True
        )
        return text_input.input_ids.to(self.device)


    def img2latent(self, image):
        """
        Converts an image to its latent representation using the CompVis VAE.

        Args:
            image: A torch tensor of shape (C, H, W), normalized to [-1, 1].

        Returns:
            x0: Latent representation of the image.
        """
        with torch.no_grad():
            # Ensure the input image has batch dimension and is on the correct device
            img_input = image.unsqueeze(0).to(self.device)

            # Encode the image using the VAE
            posterior = self.vae.encode(img_input)  # Get the posterior distribution

            # Take the mean of the posterior (latent mean)
            x0 = posterior.sample()  # Optionally use .sample() instead of .mean

            # Scale by the latent scaling factor (0.18215 for Stable Diffusion)
            x0 *= 0.18215

        return x0
    
    def id2embedding(self,input_ids):
        input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(self.tokenizer.get_vocab())).float()
        input_one_hot = torch.unsqueeze(input_one_hot,0).to(self.device)
        input_embeds = input_one_hot @ self.all_embeddings
        return input_embeds
    
    def sample(
        self,
        input_ids,
        text_embeddings,
        batch_size=1,
        seed=0,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5
    ):
        """
        A CompVis-based 'sample' method that replicates your Diffusers approach:
        1) Prepare latents
        2) For each timestep, do unconditional/conditional forward passes
        3) Apply custom SLD logic if desired
        4) Decode latents via VAE
        """

        text_embeddings = self.custom_text_encoder(input_ids = input_ids, inputs_embeds=text_embeddings)[0]
        # ------------------------------------------------------
        # 0) Encode the text or use custom embeddings
        # ------------------------------------------------------
        # For negative prompt or empty prompt
        if self.negative_prompt is not None:
            uncond_input = self.tokenizer(
                [self.negative_prompt]*batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
        else:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
        uncond_embeddings = self.id2embedding(uncond_input.input_ids.to(self.device))
        uncond_embeddings = self.custom_text_encoder(
            input_ids=uncond_input.input_ids.to(self.device),
            inputs_embeds=uncond_embeddings
        )[0]

        # Generate safety concept embeddings if needed
        if self.sld is not None:
            if self.sld_concept is not None:
                safety_concept = str(self.sld_concept)
            else:
                safety_concept = (
                    "hate, harassment, violence, suffering, humiliation, harm, suicide, "
                    "sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, "
                    "drug use, theft, vandalism, weapons, child abuse, brutality, cruelty"
                )
            safety_input = self.tokenizer(
                [safety_concept] * batch_size,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            safety_embeddings = self.id2embedding(safety_input.input_ids.to(self.device))
            safety_embeddings = self.custom_text_encoder(
                input_ids=safety_input.input_ids.to(self.device),
                inputs_embeds=safety_embeddings
            )[0]
        else:
            safety_embeddings = None

        # ------------------------------------------------------
        # 1) Initialize latents
        # ------------------------------------------------------
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.device)


        # ------------------------------------------------------
        # 2) Get alpha / sigma schedules from a DDIMSampler
        # ------------------------------------------------------

        ddim_sampler = self.sampler  # or DDIMSampler(self.model) if not set yet
        ddim_sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0)

        sigma_init = self.sampler.ddim_sigmas[-1]  # or [0], depending on the schedule

        latents = latents * sigma_init

        ddim_sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0)

        timesteps = ddim_sampler.ddim_timesteps  # ascending order
        timesteps = np.flip(timesteps,0)  # now in descending order

        alphas = ddim_sampler.ddim_alphas  # shape [num_inference_steps]
        sigmas = ddim_sampler.ddim_sigmas  # shape [num_inference_steps]

        # For SLD hyperparameters
        safety_momentum = None
        if self.sld == "weak":
            sld_warmup_steps = 15
            sld_guidance_scale = 200
            sld_threshold = 0.0
            sld_momentum_scale = 0.0
            sld_mom_beta = 0.0
        elif self.sld == "medium":
            sld_warmup_steps = 10
            sld_guidance_scale = 1000
            sld_threshold = 0.01
            sld_momentum_scale = 0.3
            sld_mom_beta = 0.4
        elif self.sld == "strong":
            sld_warmup_steps = 7
            sld_guidance_scale = 2000
            sld_threshold = 0.025
            sld_momentum_scale = 0.5
            sld_mom_beta = 0.7
        elif self.sld == "max":
            sld_warmup_steps = 0
            sld_guidance_scale = 5000
            sld_threshold = 1.0
            sld_momentum_scale = 0.5
            sld_mom_beta = 0.7

        # ------------------------------------------------------
        # 3) Sampling loop over timesteps
        # ------------------------------------------------------
        for i, step in enumerate(tqdm(timesteps)):
            # Convert step index to [0..num_inference_steps-1]
            # so we can get alpha, sigma from the right index
            j = num_inference_steps - i - 1  # if step=999 is first, j=49 if total=50
            alpha = alphas[j]
            sigma = sigmas[j]

            latent_model_input = latents * alpha.sqrt()

            timesteps = torch.tensor([step], dtype=torch.long, device=latent_model_input.device)
            # 3.2) U-Net forward passes
            with torch.no_grad():
                # Unconditional pass

                noise_pred_uncond = self.target_unet_sd(
                    latent_model_input,  # Input latents at timestep `t`
                    timesteps=timesteps,         # Current timestep 
                    context=uncond_embeddings  # Unconditional cross-attention conditioning
                )

                noise_pred_text = self.target_unet_sd(
                    latent_model_input,
                    timesteps=timesteps,
                    context=text_embeddings  # Conditional cross-attention conditioning
                )
                
            if self.sld is not None:
                noise_guidance = noise_pred_text - noise_pred_uncond

                with torch.no_grad():
                    noise_pred_safety_concept = self.target_unet_sd(latent_model_input, timesteps=timesteps, context=safety_embeddings)
                
                if safety_momentum is None:
                    safety_momentum = torch.zeros_like(noise_pred_text)

                # Equation 6
                scale = torch.clamp(
                    torch.abs((noise_pred_text - noise_pred_safety_concept)) * sld_guidance_scale, max=1.)

                # Equation 6
                safety_concept_scale = torch.where(
                    (noise_pred_text - noise_pred_safety_concept) >= sld_threshold,
                    torch.zeros_like(scale), scale)

                # Equation 4
                noise_guidance_safety = torch.mul(
                    (noise_pred_safety_concept - noise_pred_uncond), safety_concept_scale)

                # Equation 7
                noise_guidance_safety = noise_guidance_safety + sld_momentum_scale * safety_momentum

                # Equation 8
                safety_momentum = sld_mom_beta * safety_momentum + (1 - sld_mom_beta) * noise_guidance_safety

                if step >= sld_warmup_steps: # Warmup
                    # Equation 3
                    noise_guidance = noise_guidance - noise_guidance_safety
                
                noise_pred = noise_pred_uncond +  guidance_scale * noise_guidance
                
            else:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


            # Minimal manual approach:
            if j > 0:  # if not the last step
                # sample epsilon
                noise = torch.randn_like(latents, device=self.device)
            else:
                noise = torch.zeros_like(latents, device=self.device)

            a_t = alpha
            a_prev = alphas[j-1] if j > 0 else alphas[j]
            sigma_t = sigma

            # x_{t-1}
            pred_x0 = (latent_model_input - torch.sqrt(1 - a_t) * noise_pred) / torch.sqrt(a_t)
            dir_xt = torch.sqrt(1 - a_prev) * noise_pred
            latents = torch.sqrt(a_prev) * pred_x0 + dir_xt
            if sigma_t > 0:
                latents = latents + sigma_t * noise
 
        latents = latents / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latents)

        # Post-process & convert to uint8
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return images[0]


    def get_loss(self, *args, **kwargs):
        # implement your fine-tuning/adversarial logic if needed
        pass
