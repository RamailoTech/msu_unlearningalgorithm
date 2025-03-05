#mu/algorithms/erase_diff/evaluator.py

import sys
import os
import glob
import logging
import timm
import json

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from typing import Any, Dict

from mu.datasets.constants import *
from evaluation.core import BaseEvaluator
from mu.algorithms.erase_diff import EraseDiffSampler
from mu.algorithms.erase_diff.configs import ErasediffEvaluationConfig

from models import stable_diffusion
sys.modules['stable_diffusion'] = stable_diffusion

from stable_diffusion.constants.const import theme_available, class_available
from mu.datasets.constants.i2p_const import i2p_categories
from evaluation.evaluators.fid import calculate_fid_score
from mu.helpers.utils import load_categories
# from evaluation.evaluators.mu_fid import load_style_generated_images,load_style_ref_images,calculate_fid, tensor_to_float
from evaluation.evaluators.mu_fid import tensor_to_float



class EraseDiffEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
    """

    def __init__(self,config: ErasediffEvaluationConfig, **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., EraseDiffSampler).
            config (Dict[str, Any]): A dict of hyperparameters / evaluation settings.
            **kwargs: Additional overrides for config.
        """
        super().__init__(config, **kwargs)
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        self._parse_config()
        config.validate_config()
        self.config = config.to_dict()
        self.sampler = EraseDiffSampler(self.config)
        self.device = self.config['devices'][0]
        self.use_sample = self.config.get('use_sample')
        self.dataset_type = self.config.get('dataset_type')
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
        ckpt_path = self.config["classifier_ckpt_path"]
        self.logger.info(f"Loading classification checkpoint from: {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["model_state_dict"])
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
        This method now supports both the original datasets and the i2p dataset.
        For the i2p dataset, the config should specify task="i2p" and provide a list of categories.
        """
        self.logger.info("Starting accuracy calculation...")

        # Pull relevant config parameters
        input_dir = self.config['sampler_output_dir']
        output_dir = self.config["eval_output_dir"]
        seed_list = self.config.get("seed_list", [188, 288, 588, 688, 888])
        dry_run = self.config.get("dry_run", False)
        task = self.config['task']

        # For the original datasets, input_dir might be modified based on theme, etc.
        if task in ["style", "class"]:
            theme = self.config.get("forget_theme", None)
            if theme is not None:
                input_dir = os.path.join(input_dir, theme)
            else:
                input_dir = os.path.join(input_dir)

        os.makedirs(output_dir, exist_ok=True)
        self.eval_output_path = os.path.join(
            output_dir, 
            f"{self.config.get('forget_theme', 'result')}.json" if task in ["style", "class"] 
            else os.path.join(output_dir, "result.json")
        )

        # Initialize results dictionary based on task.
        if self.dataset_type == "unlearncanvas":
            if task == "style":
                # Original style task processing
                self.results = {
                    "test_theme": self.config.get("forget_theme", "sd"),
                    "input_dir": input_dir,
                    "loss": {th: 0.0 for th in theme_available},
                    "acc": {th: 0.0 for th in theme_available},
                    "pred_loss": {th: 0.0 for th in theme_available},
                    "misclassified": {th: {oth: 0 for oth in theme_available} for th in theme_available}
                }
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
                            self.results["pred_loss"][test_theme] += pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val
                            # Normalize accuracy by number of images processed per theme.
                            self.results["acc"][test_theme] += (pred_success * 1.0 / (len(class_available) * len(seed_list)))
                            misclassified_as = theme_available[pred_label.item()]
                            self.results["misclassified"][test_theme][misclassified_as] += 1

                    if not dry_run:
                        self.save_results()

            elif task == "class":
                # Original class task processing
                self.results = {
                    "loss": {cls_: 0.0 for cls_ in class_available},
                    "acc": {cls_: 0.0 for cls_ in class_available},
                    "pred_loss": {cls_: 0.0 for cls_ in class_available},
                    "misclassified": {cls_: {other_cls: 0 for other_cls in class_available} for cls_ in class_available}
                }
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
                                loss = F.cross_entropy(res, label)
                                res_softmax = F.softmax(res, dim=1)
                                pred_loss_val = res_softmax[0][label_val]
                                pred_label = torch.argmax(res)
                                pred_success = (pred_label == label_val).sum()

                            # Accumulate stats
                            self.results["loss"][object_class] += loss.item()
                            self.results["pred_loss"][object_class] += pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val
                            self.results["acc"][object_class] += (pred_success * 1.0 / (len(class_available) * len(seed_list)))
                            misclassified_as = class_available[pred_label.item()]
                            self.results["misclassified"][object_class][misclassified_as] += 1

                    if not dry_run:
                        self.save_results()

        elif self.dataset_type == "i2p":

            categories = i2p_categories
            self.results = {
                "loss": {cat: 0.0 for cat in categories},
                "acc": {cat: 0.0 for cat in categories},
                "pred_loss": {cat: 0.0 for cat in categories},
                "misclassified": {cat: {other_cat: 0 for other_cat in categories} for cat in categories},
                "input_dir": input_dir
            }
            # Iterate over each category and each seed value.
            for category in tqdm(categories, total=len(categories)):
                for seed in seed_list:
                    # Construct filename; adjust the naming convention if necessary.
                    img_file = f"{category}_seed_{seed}.jpg"
                    img_path = os.path.join(input_dir, img_file)
                    if not os.path.exists(img_path):
                        self.logger.warning(f"Image not found: {img_path}")
                        continue

                    # Preprocess and forward pass.
                    image = Image.open(img_path)
                    tensor_img = self.preprocess_image(image)
                    # Ground-truth label is the index of the category.
                    label_idx = categories.index(category)
                    label = torch.tensor([label_idx]).to(self.device)
                    with torch.no_grad():
                        res = self.model(tensor_img)
                        loss = F.cross_entropy(res, label)
                        res_softmax = F.softmax(res, dim=1)
                        pred_loss_val = res_softmax[0][label_idx]
                        pred_label = torch.argmax(res)
                        pred_success = (pred_label == label_idx).sum()

                    # Accumulate statistics.
                    self.results["loss"][category] += loss.item()
                    self.results["pred_loss"][category] += pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val
                    # Here, we assume the accuracy is averaged over the number of seeds per category.
                    self.results["acc"][category] += (pred_success * 1.0 / len(seed_list))
                    misclassified_as = categories[pred_label.item()]
                    self.results["misclassified"][category][misclassified_as] += 1

                if not dry_run:
                    self.save_results()

        
        elif self.dataset_type == "generic":

            categories = load_categories(self.config["reference_dir"])

            self.results = {
                "loss": {cat: 0.0 for cat in categories},
                "acc": {cat: 0.0 for cat in categories},
                "pred_loss": {cat: 0.0 for cat in categories},
                "misclassified": {cat: {other_cat: 0 for other_cat in categories} for cat in categories},
                "input_dir": input_dir
            }
            # Iterate over each category and each seed value.
            for category in tqdm(categories, total=len(categories)):
                for seed in seed_list:
                    # Construct filename; adjust the naming convention if necessary.
                    img_file = f"{category}_seed_{seed}.jpg"
                    img_path = os.path.join(input_dir, img_file)
                    if not os.path.exists(img_path):
                        self.logger.warning(f"Image not found: {img_path}")
                        continue

                    # Preprocess and forward pass.
                    image = Image.open(img_path)
                    tensor_img = self.preprocess_image(image)
                    # Ground-truth label is the index of the category.
                    label_idx = categories.index(category)
                    label = torch.tensor([label_idx]).to(self.device)
                    with torch.no_grad():
                        res = self.model(tensor_img)
                        loss = F.cross_entropy(res, label)
                        res_softmax = F.softmax(res, dim=1)
                        pred_loss_val = res_softmax[0][label_idx]
                        pred_label = torch.argmax(res)
                        pred_success = (pred_label == label_idx).sum()

                    # Accumulate statistics.
                    self.results["loss"][category] += loss.item()
                    self.results["pred_loss"][category] += pred_loss_val.item() if isinstance(pred_loss_val, torch.Tensor) else pred_loss_val
                    # Here, we assume the accuracy is averaged over the number of seeds per category.
                    self.results["acc"][category] += (pred_success * 1.0 / len(seed_list))
                    misclassified_as = categories[pred_label.item()]
                    self.results["misclassified"][category][misclassified_as] += 1

                if not dry_run:
                    self.save_results()

        else:
            self.logger.error(f"Unknown dataset type: {self.dataset_type}")
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.logger.info("Accuracy calculation completed.")


    def calculate_fid_score(self, *args, **kwargs):
        """
        Calculate the Fréchet Inception Distance (FID) score using the images 
        generated by EraseDiffSampler vs. some reference images. 
        """
        self.logger.info("Starting FID calculation...")
        # self.theme_available = uc_sample_theme_available_eval if self.use_sample else uc_theme_available
        # self.class_available = uc_sample_class_available_eval if self.use_sample else uc_class_available
        generated_path = self.config["sampler_output_dir"]  
        
        if self.dataset_type in ("i2p", "generic"):
            reference_path = f"{self.config['reference_dir']}/images"
        elif self.dataset_type == "unlearncanvas":
            reference_path = self.config['reference_dir']

        fid_value, _ = calculate_fid_score(generated_path, reference_path)  
        self.results["FID"] = fid_value

        # forget_theme = self.config.get("forget_theme", None) 
        # use_multiprocessing = self.config.get("multiprocessing", False)
        # batch_size = self.config.get("batch_size", 64)

        # images_generated = load_style_generated_images(
        #     path=generated_path, 
        #     theme_available=self.theme_available,
        #     class_available=self.class_available,
        #     exclude=forget_theme, 
        #     seed=self.config.get("seed_list", [188, 288, 588, 688, 888])
        # )
        # images_reference = load_style_ref_images(
        #     path=reference_path, 
        #     theme_available=self.theme_available,
        #     class_available=self.class_available,
        #     use_sample = self.use_sample,
        #     exclude=forget_theme
        # )

        # fid_value = calculate_fid(
        #     images1=images_reference, 
        #     images2=images_generated, 
        #     use_multiprocessing=use_multiprocessing, 
        #     batch_size=batch_size
        # )
        # self.logger.info(f"Calculated FID: {fid_value}")
        # self.results["FID"] = fid_value

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
        self.sampler.load_model()  
        self.sampler.sample()    

        # Load the classification model
        self.load_model()

        # Proceed with accuracy and FID calculations
        self.calculate_accuracy()
        self.calculate_fid_score()

        # Save results
        self.save_results()

        self.logger.info("Evaluation run completed.")

