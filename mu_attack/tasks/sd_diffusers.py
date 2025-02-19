# mu_attack/tasks/sd_diffusers.py

import torch
from tqdm.auto import tqdm
from copy import deepcopy
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    LMSDiscreteScheduler,
)
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_attack.tasks.utils.metrics.style_eval import init_classifier
from mu_attack.tasks.utils.metrics.object_eval import imagenet_ResNet50
from mu_attack.tasks.utils.metrics.harm_eval import q16_binary_classifier
from mu_attack.core import BaseStableDiffusionPipeline


class BaseDiffusersPipeline(BaseStableDiffusionPipeline):
    
    def __init__(self, model_name_or_path,cache_path, device, concept,  sld, sld_concept, negative_prompt, target_ckpt, criterion, classifier_dir,model_name=None,*args,**kwargs):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.cache_path = cache_path
        self.sld = sld 
        self.concept= concept 
        self.sld_concept = sld_concept
        self.negative_prompt = negative_prompt
        self.target_ckpt = target_ckpt
        self.classifier_dir = classifier_dir
        self.model_name = model_name 
        self.criterion = torch.nn.L1Loss() if criterion == 'l1' else torch.nn.MSELoss()
        
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

        if self.model_name == 'SD-v1-4':
            self.dir_ = "CompVis/stable-diffusion-v1-4"
        elif self.model_name == 'SD-V2':
            self.dir_ = "stabilityai/stable-diffusion-2-base"
        elif self.model_name == 'SD-V2-1':
            self.dir_ = "stabilityai/stable-diffusion-2-1-base"
        else:
            self.dir_ = "CompVis/stable-diffusion-v1-4"

    def load_model(self):
        if self.target_ckpt is not None:
            self.vae = AutoencoderKL.from_pretrained(self.dir_, subfolder="vae").to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.dir_, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.dir_, subfolder="text_encoder").to(self.device)
            self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
            self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)
            self.unet_sd = UNet2DConditionModel.from_pretrained(self.dir_, subfolder="unet").to(self.device)
            self.target_unet_sd = deepcopy(self.unet_sd)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name_or_path, subfolder="tokenizer", cache_dir=self.cache_path)
            self.text_encoder = CLIPTextModel.from_pretrained(self.model_name_or_path, subfolder="text_encoder", cache_dir=self.cache_path).to(self.device)
            self.vae = AutoencoderKL.from_pretrained(self.model_name_or_path,subfolder="vae", cache_dir=self.cache_path).to(self.device)
            self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
            self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)
            self.unet_sd = UNet2DConditionModel.from_pretrained(self.model_name_or_path, subfolder="unet", cache_dir=self.cache_path).to(self.device)
            self.target_unet_sd = deepcopy(self.unet_sd)

        if self.sld is None:
            self.target_unet_sd.load_state_dict(torch.load(self.target_ckpt, map_location=self.device))
        if self.classifier_dir is not None:
            self.classifier = init_classifier(self.device,self.classifier_dir)
        elif self.concept in self.object_list:
            self.processor, self.classifier = imagenet_ResNet50(self.device)
        elif self.concept == 'harm':
            self.clip_model, self.classifier = q16_binary_classifier(self.device)
        
        # e.g. LMS or DDIM, depending on user preference
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            num_train_timesteps=1000
        )

        for m in [self.vae, self.text_encoder, self.custom_text_encoder, self.unet_sd, self.target_unet_sd]:
            m.eval()
            m.requires_grad_(False)

    def str2id(self,prompt):
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt", truncation=True
        )
        return text_input.input_ids.to(self.device)
    
    def img2latent(self,image):
        with torch.no_grad():
            img_input  = image.unsqueeze(0).to(self.device)
            x0 = self.vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        return x0
    
    def id2embedding(self,input_ids):
        input_one_hot = F.one_hot(input_ids.view(-1), num_classes = len(self.tokenizer.get_vocab())).float()
        input_one_hot = torch.unsqueeze(input_one_hot,0).to(self.device)
        input_embeds = input_one_hot @ self.all_embeddings
        return input_embeds
    
    def sample(self,input_ids,text_embeddings,batch_size=1,seed=0,height=512,width=512,num_inference_steps=50,guidance_scale=7.5):
        
        text_embeddings = self.custom_text_encoder(input_ids = input_ids, inputs_embeds=text_embeddings)[0]
        
        # Negative prompt or Not
        if self.negative_prompt is not None:
            uncond_input = self.tokenizer(
                [self.negative_prompt] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
        else:
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
        
        uncond_embeddings = self.id2embedding(uncond_input.input_ids.to(self.device))
        uncond_embeddings = self.custom_text_encoder(input_ids = uncond_input.input_ids.to(self.device), inputs_embeds=uncond_embeddings)[0]
        
        # Generate safety embedding for SLD
        if self.sld is not None:
            if self.sld_concept is not None:
                safety_concept = str(self.sld_concept)
            else:
                safety_concept = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
                                            'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, ' \
                                            'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'

            safety_input = self.tokenizer(
                [safety_concept] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            
            safety_embeddings = self.id2embedding(safety_input.input_ids.to(self.device))
            safety_embeddings = self.custom_text_encoder(input_ids = safety_input.input_ids.to(self.device), inputs_embeds=safety_embeddings)[0]
            
        
        generator = torch.manual_seed(seed)  
        latents = torch.randn(
            (batch_size, self.target_unet_sd.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)
        
        # Hyperparameters for different SLD Modes
        safety_momentum = None
        if self.sld == 'weak':
            sld_warmup_steps = 15       # SLD warmup steps
            sld_guidance_scale = 200    # SLD gudidance Scale
            sld_threshold = 0.0         # SLD threshold
            sld_momentum_scale = 0.0    # SLD momentum scale
            sld_mom_beta = 0.0          # SLD momentum beta
        elif self.sld == 'medium':
            sld_warmup_steps = 10       # SLD warmup steps
            sld_guidance_scale = 1000   # SLD gudidance Scale
            sld_threshold = 0.01        # SLD threshold
            sld_momentum_scale = 0.3    # SLD momentum scale
            sld_mom_beta = 0.4          # SLD momentum beta
        elif self.sld == 'strong':
            sld_warmup_steps = 7         # SLD warmup steps
            sld_guidance_scale = 2000    # SLD gudidance Scale
            sld_threshold = 0.025        # SLD threshold
            sld_momentum_scale = 0.5     # SLD momentum scale
            sld_mom_beta = 0.7           # SLD momentum beta
        elif self.sld == 'max':
            sld_warmup_steps = 0         # SLD warmup steps
            sld_guidance_scale = 5000    # SLD gudidance Scale
            sld_threshold = 1.0          # SLD threshold
            sld_momentum_scale = 0.5     # SLD momentum scale
            sld_mom_beta = 0.7           # SLD momentum beta

        for t in tqdm(self.scheduler.timesteps):
            
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred_uncond = self.target_unet_sd(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample
                noise_pred_text = self.target_unet_sd(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # Perform SLD guidance
            if self.sld is not None:
                noise_guidance = noise_pred_text - noise_pred_uncond
                
                with torch.no_grad():
                    noise_pred_safety_concept = self.target_unet_sd(latent_model_input, t, encoder_hidden_states=safety_embeddings).sample
                
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

                if t >= sld_warmup_steps: # Warmup
                    # Equation 3
                    noise_guidance = noise_guidance - noise_guidance_safety
                
                noise_pred = noise_pred_uncond +  guidance_scale * noise_guidance
                
            else:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images[0]
    
    def get_loss(self, *args, **kwargs):
        pass
