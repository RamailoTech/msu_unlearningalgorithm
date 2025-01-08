import os
import logging
import torch
import numpy as np
from PIL import Image
from torch import autocast
from mu.core.base_sampler import BaseSampler        
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.constants.const import theme_available, class_available
from mu.helpers import load_config
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
from mu.helpers.utils import load_ckpt_from_config,load_style_generated_images,load_style_ref_images,calculate_fid
import timm
from tqdm import tqdm
from typing import Any, Dict
from torchvision import transforms
from torch.nn import functional as F
from mu.core.base_evaluator import BaseEvaluator

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
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
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
        batch_size = self.config["num_samples"]       
        cfg_text = self.config["cfg_text"]    
        seed = self.config["seed"]
        height = self.config["image_height"]
        width = self.config["image_width"]
        ddim_eta = self.config["ddim_eta"]
        output_dir = self.config["sampler_output_dir"]

        os.makedirs(output_dir,theme, exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")
       
        for test_theme in theme_available:
            for object_class in class_available:
                output_path = f"{output_dir}/{test_theme}_{object_class}_seed{seed}.jpg"
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



class UnifiedConceptEditingEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
    """

    def __init__(self,config: Dict[str, Any], **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., UnifiedConceptEditingSampler).
            config (Dict[str, Any]): A dict of hyperparameters / evaluation settings.
            **kwargs: Additional overrides for config.
        """
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampler = UnifiedConceptEditingSampler(config)
        self.classification_model = None
        self.results = {}

    def load_model(self, *args, **kwargs):
        """
        Load the classification model for evaluation, using 'timm' 
        or any approach you prefer. 
        We assume your config has 'classification_ckpt' and 'task' keys, etc.
        """
        self.logger.info("Loading classification model...")
        model = self.config.get("classification_model")
        self.classification_model = timm.create_model(
            model, 
            pretrained=True
        ).to(self.device)
        task = self.config['task'] # "style" or "class"
        num_classes = len(theme_available) if task == "style" else len(class_available)
        self.classification_model.head = torch.nn.Linear(1024, num_classes).to(self.device)

        # Load checkpoint
        ckpt_path = self.config["classification_ckpt"]
        self.logger.info(f"Loading classification checkpoint from: {ckpt_path}")
        #NOTE: changed model_state_dict to state_dict as it was not present and added strict=False
        self.classification_model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["state_dict"],strict=False)
        self.classification_model.eval()
    
        self.logger.info("Classification model loaded successfully.")

    def preprocess_image(self, image: Image.Image):
        """
        Preprocess the input PIL image before feeding into the classifier.
        Replicates the transforms from your accuracy.py script.
        """
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return image_transform(image).unsqueeze(0).to(self.device)

    def calculate_accuracy(self, *args, **kwargs):
        """
        Calculate accuracy of the classification model on generated images.
        Mirrors the logic from your accuracy.py but integrated into a single method.
        """
        self.logger.info("Starting accuracy calculation...")

        # Pull relevant config
        theme = self.config.get("theme", None)
        input_dir = self.config['sampler_output_dir']
        output_dir = self.config["eval_output_dir"]
        seed_list = self.config.get("seed_list", [188, 288, 588, 688, 888])
        dry_run = self.config.get("dry_run", False)
        task = self.config['task']  

        if theme is not None:
            input_dir = os.path.join(input_dir, theme)
        else:
            input_dir = os.path.join(input_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = (os.path.join(output_dir, f"{theme}.pth") 
                       if theme is not None 
                       else os.path.join(output_dir, "result.pth"))

        # Initialize results dictionary
        self.results = {
            "test_theme": theme if theme is not None else "sd",
            "input_dir": input_dir,
        }

        if task == "style":
            self.results["loss"] = {th: 0.0 for th in theme_available}
            self.results["acc"] = {th: 0.0 for th in theme_available}
            self.results["pred_loss"] = {th: 0.0 for th in theme_available}
            self.results["misclassified"] = {
                th: {oth: 0 for oth in theme_available} 
                for th in theme_available
            }
        else:  # task == "class"
            self.results["loss"] = {cls_: 0.0 for cls_ in class_available}
            self.results["acc"] = {cls_: 0.0 for cls_ in class_available}
            self.results["pred_loss"] = {cls_: 0.0 for cls_ in class_available}
            self.results["misclassified"] = {
                cls_: {other_cls: 0 for other_cls in class_available} 
                for cls_ in class_available
            }

        # Evaluate
        if task == "style":
            for idx, test_theme in tqdm(enumerate(theme_available), total=len(theme_available)):
                theme_label = idx
                for seed in seed_list:
                    for object_class in class_available:
                        img_file = f"{test_theme}_{object_class}_seed{seed}.jpg"
                        img_path = os.path.join(input_dir, img_file)
                        if not os.path.exists(img_path):
                            self.logger.warning(f"Image not found: {img_path}")
                            continue

                        # Preprocess
                        image = Image.open(img_path)
                        tensor_img = self.preprocess_image(image)
                        label = torch.tensor([theme_label]).to(self.device)

                        # Forward pass
                        with torch.no_grad():
                            res = self.classification_model(tensor_img)

                        # Compute losses
                        loss = F.cross_entropy(res, label)
                        res_softmax = F.softmax(res, dim=1)
                        pred_loss_val = res_softmax[0][theme_label].item()
                        pred_label = torch.argmax(res).item()
                        pred_success = (pred_label == theme_label)

                        # Accumulate stats
                        self.results["loss"][test_theme] += loss.item()
                        self.results["pred_loss"][test_theme] += pred_loss_val
                        # Probability of success is 1 if pred_success else 0,
                        # but for your code, you were dividing by total. So let's keep a sum for now:
                        self.results["acc"][test_theme] += (1 if pred_success else 0)

                        misclassified_as = theme_available[pred_label]
                        self.results["misclassified"][test_theme][misclassified_as] += 1

                if not dry_run:
                    self.save_results(self.results, output_path)

        else: # task == "class"
            for test_theme in tqdm(theme_available, total=len(theme_available)):
                for seed in seed_list:
                    for idx, object_class in enumerate(class_available):
                        label_val = idx
                        img_file = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                        img_path = os.path.join(input_dir, img_file)
                        if not os.path.exists(img_path):
                            self.logger.warning(f"Image not found: {img_path}")
                            continue

                        # Preprocess
                        image = Image.open(img_path)
                        tensor_img = self.preprocess_image(image)
                        label = torch.tensor([label_val]).to(self.device)

                        with torch.no_grad():
                            res = self.classification_model(tensor_img)

                        loss = F.cross_entropy(res, label)
                        res_softmax = F.softmax(res, dim=1)
                        pred_loss_val = res_softmax[0][label_val].item()
                        pred_label = torch.argmax(res).item()
                        pred_success = (pred_label == label_val)

                        self.results["loss"][object_class] += loss.item()
                        self.results["pred_loss"][object_class] += pred_loss_val
                        self.results["acc"][object_class] += (1 if pred_success else 0)

                        misclassified_as = class_available[pred_label]
                        self.results["misclassified"][object_class][misclassified_as] += 1

                if not dry_run:
                    self.save_results(self.results, output_path)

        self.logger.info("Accuracy calculation completed.")

    def calculate_fid_score(self, *args, **kwargs):
        """
        Calculate the Fréchet Inception Distance (FID) score using the images 
        generated by SaliencyUnlearningSampler vs. some reference images. 
        """
        self.logger.info("Starting FID calculation...")

        generated_path = self.config["sampler_output_dir"]  
        reference_path = self.config["reference_dir"]       
        forget_theme = self.config.get("forget_theme", None) 
        use_multiprocessing = self.config.get("multiprocessing", False)
        batch_size = self.config.get("batch_size", 64)
        output_dir = self.config["fid_output_path"]
        os.makedirs(output_dir, exist_ok=True)

        images_generated = load_style_generated_images(
            path=generated_path, 
            exclude=forget_theme, 
            seed=self.config.get("seed_list", [188, 288, 588, 688, 888])
        )
        images_reference = load_style_ref_images(
            path=reference_path, 
            exclude=forget_theme
        )

        fid_value = calculate_fid(
            images1=images_reference, 
            images2=images_generated, 
            use_multiprocessing=use_multiprocessing, 
            batch_size=batch_size
        )
        self.logger.info(f"Calculated FID: {fid_value}")
        self.results["FID"] = fid_value
        fid_path = os.path.join(output_dir, "fid_value.pth")
        torch.save({"FID": fid_value}, fid_path)
        self.logger.info(f"FID results saved to: {fid_path}")

    def save_results(self, results: dict, output_path: str):
        """
        Save evaluation results to a file. You can also do JSON or CSV if desired.
        """
        torch.save(results, output_path)
        self.logger.info(f"Results saved to: {output_path}")

    def run(self, *args, **kwargs):
        """
        Run the complete evaluation process:
        1) Load the classification model
        2) Generate images (using sampler)
        3) Calculate accuracy
        4) Calculate FID
        5) Save final results
        """

        # Call the sample method to generate images
        self.sampler.load_model()  
        self.sampler.sample()    

        # Load the classification model
        self.load_model()

        # Proceed with accuracy and FID calculations
        self.calculate_accuracy()
        self.calculate_fid_score()

        # Save results
        self.save_results(self.results, os.path.join(self.config["eval_output_dir"], "final_results.pth"))

        self.logger.info("Evaluation run completed.")

