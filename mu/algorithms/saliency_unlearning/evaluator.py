#mu/algorithms/saliency_unlearning/evaluator.py

import os
import logging
import torch
import timm
import json
from tqdm import tqdm
from typing import Any, Dict

from PIL import Image  
from torchvision import transforms
from torch.nn import functional as F

from mu.core.base_evaluator import BaseEvaluator
from mu.algorithms.saliency_unlearning import SaliencyUnlearningSampler
from stable_diffusion.constants.const import theme_available, class_available
from mu.helpers.utils import load_style_generated_images,load_style_ref_images,calculate_fid, tensor_to_float

#TODO remove this
theme_available = ['Abstractionism', 'Bricks', 'Cartoon']
class_available = ['Architectures', 'Bears', 'Birds']


class SaliencyUnlearningEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
    """

    def __init__(self,config: Dict[str, Any], **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., SaliencyUnlearningSampler).
            config (Dict[str, Any]): A dict of hyperparameters / evaluation settings.
            **kwargs: Additional overrides for config.
        """
        super().__init__(config, **kwargs)
        self.config = config
        self.sampler = SaliencyUnlearningSampler(config)
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
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["state_dict"],strict=False)
        self.model.eval()
    
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
        theme = self.config.get("forget_theme", None)
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
        self.eval_output_path = (os.path.join(output_dir, f"{theme}.json") 
                       if theme is not None 
                       else os.path.join(output_dir, "result.json"))

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

                        # Forward pass
                        with torch.no_grad():
                            res = self.model(tensor_img)
                            label = torch.tensor([theme_label]).to(self.device)
                            loss = F.cross_entropy(res, label)

                            # Compute losses
                            res_softmax = F.softmax(res, dim=1)
                            pred_loss_val = res_softmax[0][theme_label]
                            pred_label = torch.argmax(res)
                            pred_success = (pred_label == theme_label).sum()

                        # Accumulate stats
                        self.results["loss"][test_theme] += loss.item()
                        self.results["pred_loss"][test_theme] += pred_loss_val
                        self.results["acc"][test_theme] += (pred_success * 1.0 / (len(class_available) * len(self.config['seed_list'])))

                        misclassified_as = theme_available[pred_label.item()]
                        self.results["misclassified"][test_theme][misclassified_as] += 1

                if not dry_run:
                    self.save_results()

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
                            res = self.model(tensor_img)
                            label = torch.tensor([label_val]).to(self.device)
                            loss = F.cross_entropy(res, label)

                            # Compute losses
                            res_softmax = F.softmax(res, dim=1)
                            pred_loss_val = res_softmax[0][label_val]
                            pred_label = torch.argmax(res)
                            pred_success = (pred_label == label_val).sum()

                        # Accumulate stats
                        self.results["loss"][object_class] += loss.item()
                        self.results["pred_loss"][object_class] += pred_loss_val
                        self.results["acc"][object_class] += (pred_success * 1.0 / (len(class_available) * len(self.config['seed_list'])))

                        misclassified_as = class_available[pred_label.item()]
                        self.results["misclassified"][object_class][misclassified_as] += 1


                if not dry_run:
                    self.save_results()

        self.logger.info("Accuracy calculation completed.")

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


    def save_results(self, *args, **kwargs):
        """
        Save whatever is present in `self.results` to a JSON file.
        """
        try:
            # Convert all tensors before saving
            converted_results = tensor_to_float(self.results)
            with open(self.eval_output_path, 'w') as json_file:
                json.dump(converted_results, json_file, indent=4)
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

