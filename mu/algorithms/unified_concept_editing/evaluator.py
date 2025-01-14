#mu/algorithms/unified_concept_editing/evaluator.py

import os
import logging
import timm
from tqdm import tqdm
from typing import Any, Dict
from PIL import Image

import torch
from torchvision import transforms
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline
from stable_diffusion.constants.const import theme_available, class_available
from mu.helpers.utils import load_style_generated_images,load_style_ref_images,calculate_fid
from mu.core.base_evaluator import BaseEvaluator
from mu.algorithms.unified_concept_editing import UnifiedConceptEditingSampler
import json

#TODO remove this
theme_available = ['Bricks']
class_available = ['Architectures', 'Bears', 'Birds']


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
        self.config = config
        self.sampler = UnifiedConceptEditingSampler(config)
        self.device = self.config['devices'][0]
        self.model = None
        self.eval_output_path = None
        self.results = {}

        self.logger = logging.getLogger(__name__)


    def load_model(self, *args, **kwargs):
        """
        Load the classification model for evaluation, using 'timm' 
        or any approach you prefer. 
        We assume your config has 'ckpt_path' and 'task' keys, etc.
        """
        self.logger.info("Loading classification model...")
        model = self.config.get("classification_model")
        self.model = timm.create_model(
            model, 
            pretrained=True
        ).to(self.device)
        task = self.config['task'] # "style" or "class"
        num_classes = len(theme_available) if task == "style" else len(class_available)
        self.model.head = torch.nn.Linear(1024, num_classes).to(self.device)

        # Load checkpoint
        ckpt_path = self.config["model_ckpt_path"]
        self.logger.info(f"Loading classification checkpoint from: {ckpt_path}")
        #NOTE: changed model_state_dict to state_dict as it was not present and added strict=False
        # self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device),strict=False)
        self.model = StableDiffusionPipeline.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32
        ).to(self.device)
        # self.model.eval()
    
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
        Calculate accuracy of the diffusion model on generated images using Stable Diffusion outputs.
        """
        self.logger.info("Starting accuracy calculation for Stable Diffusion model...")

        # Extract relevant configurations
        theme = self.config.get("theme", None)
        output_dir = self.config["eval_output_dir"]
        seed_list = self.config.get("seed_list", [188, 288, 588, 688, 888])
        dry_run = self.config.get("dry_run", False)
        task = self.config['task']

        os.makedirs(output_dir, exist_ok=True)
        self.eval_output_path = os.path.join(output_dir, f"{theme}.json" if theme else "result.json")

        # Initialize results dictionary
        self.results = {
            "test_theme": theme if theme is not None else "sd",
            "input_dir": "generated_by_diffusion_model",
        }

        # Prepare result structures for style and class tasks
        if task == "style":
            self.results["loss"] = {th: 0.0 for th in theme_available}
            self.results["acc"] = {th: 0.0 for th in theme_available}
            self.results["misclassified"] = {
                th: {oth: 0 for oth in theme_available} 
                for th in theme_available
            }
        else:  # task == "class"
            self.results["loss"] = {cls_: 0.0 for cls_ in class_available}
            self.results["acc"] = {cls_: 0.0 for cls_ in class_available}
            self.results["misclassified"] = {
                cls_: {other_cls: 0 for other_cls in class_available} 
                for cls_ in class_available
            }

        # Begin the generation and evaluation process
        if task == "style":
            for idx, test_theme in tqdm(enumerate(theme_available), total=len(theme_available)):
                theme_label = idx
                for seed in seed_list:
                    try:
                        # Generate an image using the Stable Diffusion pipeline
                        with torch.no_grad():
                            generated_image = self.model(
                                prompt=f"A {test_theme} style image", 
                                generator=torch.manual_seed(seed)
                            ).images[0]

                        # Preprocess the generated image
                        tensor_img = self.preprocess_image(generated_image)

                        # Perform forward pass through the classification model
                        with torch.no_grad():
                            logits = self.model.unet(
                                tensor_img.half() if self.device.startswith('cuda') else tensor_img.float()
                            ).sample

                        # Generate label for evaluation
                        pred_label = torch.argmax(logits.mean(dim=[1, 2, 3])) if logits.ndim == 4 else theme_label
                        pred_success = int(pred_label == theme_label)

                        # Calculate loss (MSE loss used here as a proxy due to lack of logits)
                        loss = F.mse_loss(logits.float(), tensor_img.float())

                        # Accumulate results
                        self.results["loss"][test_theme] += loss.item()
                        self.results["acc"][test_theme] += pred_success / len(seed_list)
                        misclassified_as = theme_available[pred_label]
                        self.results["misclassified"][test_theme][misclassified_as] += 1

                    except Exception as e:
                        self.logger.error(f"Error during generation or classification: {e}")

                if not dry_run:
                    self.save_results()

        else:  # task == "class"
            for test_theme in tqdm(theme_available, total=len(theme_available)):
                for seed in seed_list:
                    for idx, object_class in enumerate(class_available):
                        try:
                            label_val = idx

                            # Generate an image using the Stable Diffusion pipeline
                            with torch.no_grad():
                                generated_image = self.model(
                                    prompt=f"A {object_class} image", 
                                    generator=torch.manual_seed(seed)
                                ).images[0]

                            # Preprocess the generated image
                            tensor_img = self.preprocess_image(generated_image)

                            # Perform forward pass through the classification model
                            with torch.no_grad():
                                logits = self.model.unet(
                                    tensor_img.half() if self.device.startswith('cuda') else tensor_img.float()
                                ).sample

                            # Generate label and calculate accuracy
                            pred_label = torch.argmax(logits.mean(dim=[1, 2, 3])) if logits.ndim == 4 else label_val
                            pred_success = int(pred_label == label_val)

                            # Calculate loss (MSE loss as proxy)
                            loss = F.mse_loss(logits.float(), tensor_img.float())

                            # Accumulate results
                            self.results["loss"][object_class] += loss.item()
                            self.results["acc"][object_class] += pred_success / len(seed_list)
                            misclassified_as = class_available[pred_label]
                            self.results["misclassified"][object_class][misclassified_as] += 1

                        except Exception as e:
                            self.logger.error(f"Error during generation or classification: {e}")

                if not dry_run:
                    self.save_results()

        self.logger.info("Stable Diffusion model accuracy calculation completed.")


    def calculate_fid_score(self, *args, **kwargs):
        """
        Calculate the Fréchet Inception Distance (FID) score using the images 
        generated by EraseDiffSampler vs. some reference images. 
        """
        self.logger.info("Starting FID calculation...")

        generated_path = self.config["sampler_output_dir"]  
        reference_path = self.config["reference_dir"]       
        forget_theme = self.config.get("forget_theme", None) 
        use_multiprocessing = self.config.get("multiprocessing", False)
        batch_size = self.config.get("batch_size", 64)

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
        # self.eval_output_path = os.path.join(output_dir, "fid_value.pth")


    # def save_results(self,*args, **kwargs):
    #     """
    #     Save evaluation results to a file. You can also do JSON or CSV if desired.
    #     """
    #     torch.save(self.results, self.eval_output_path)
    #     self.logger.info(f"Results saved to: {self.eval_output_path}")

    def save_results(self, *args, **kwargs):
        """
        Save whatever is present in `self.results` to a JSON file.
        """
        try:
            with open(self.eval_output_path, 'w') as json_file:
                json.dump(self.results, json_file, indent=4)
            self.logger.info(f"Results saved to: {self.eval_output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to JSON file: {e}")



    def run(self, *args, **kwargs):
        """
       Run the complete evaluation process:
        1) Load the model checkpoint
        2) Generate images (using sampler)
        3) Load the classification model
        4) Calculate accuracy
        5) Calculate FID
        6) Save final results
        """

        # Call the sample method to generate images
        # self.sampler.load_model()  
        # self.sampler.sample()    

        # Load the classification model
        self.load_model()

        # Proceed with accuracy and FID calculations
        self.calculate_accuracy()
        self.calculate_fid_score()

        # Save results
        self.save_results()

        self.logger.info("Evaluation run completed.")

