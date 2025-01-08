import os
import logging
import torch
import numpy as np
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything
from mu.core.base_sampler import BaseSampler        
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.constants.const import theme_available, class_available
from mu.helpers import load_config
from mu.helpers.utils import load_ckpt_from_config,load_style_generated_images,load_style_ref_images,calculate_fid
import timm
from tqdm import tqdm
from typing import Any, Dict
from torchvision import transforms
from torch.nn import functional as F
from mu.core.base_evaluator import BaseEvaluator

theme_available = ['Abstractionism', 'Bricks', 'Cartoon']
class_available = ['Architectures', 'Bears', 'Birds']

class ScissorHandsSampler(BaseSampler):
    """ScissorHands Image Generator class extending a hypothetical BaseImageGenerator."""

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the ScissorHandsSampler with a YAML config (or dict).
        
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
        theme = self.config["theme"]         
        cfg_text = self.config["cfg_text"]    
        seed = self.config["seed"]
        H = self.config["image_height"]
        W = self.config["image_width"]
        ddim_eta = self.config["ddim_eta"]
        output_dir = self.config["sampler_output_dir"]

        os.makedirs(output_dir,theme, exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")

        seed_everything(seed)
       
        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_',' ')} style."
                self.logger.info(f"Sampling prompt: {prompt}")
                
                with torch.no_grad():
                    with autocast(self.device):
                        with self.model.ema_scope():
                            uc = self.model.get_learned_conditioning([""])  
                            c  = self.model.get_learned_conditioning(prompt)
                            shape = [4, H // 8, W // 8]
                            # Generate samples
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

                            # Convert to numpy image
                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            assert len(x_samples_ddim) == 1


                            # Convert to uint8 image
                            x_sample = x_samples_ddim[0]

                            # x_sample = (255. * x_sample.numpy()).round()
                            if isinstance(x_sample, torch.Tensor):
                                x_sample = (255. * x_sample.cpu().detach().numpy()).round()
                            else:
                                x_sample = (255. * x_sample).round()
                            x_sample = x_sample.astype(np.uint8)
                            img = Image.fromarray(x_sample)

                            #save image
                            filename = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                            outpath = os.path.join(output_dir,theme, filename)
                            self.save_image(img, outpath)

        self.logger.info("Image generation completed.")

    def save_image(self, image: Image.Image, file_path: str) -> None:
        """
        Save an image to the specified path.
        """
        image.save(file_path)
        self.logger.info(f"Image saved at: {file_path}")



class ScissorHandsEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
    """

    def __init__(self,config: Dict[str, Any], **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., ScissorHandsSampler).
            config (Dict[str, Any]): A dict of hyperparameters / evaluation settings.
            **kwargs: Additional overrides for config.
        """
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampler = ScissorHandsSampler(config)
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
        generated by ScissorHandsSampler vs. some reference images. 
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

