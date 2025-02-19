# mu_defense/algorithms/adv_unlearn/evaluator.py

import os
import pandas as pd
import json
import logging

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import models, transforms
from T2IBenchmark import calculate_fid

from mu_defense.algorithms.adv_unlearn.configs import MUDefenseEvaluationConfig
from evaluation.core import DefenseBaseEvaluator


class MUDefenseEvaluator(DefenseBaseEvaluator):
    """Evaluator for the defense."""
    
    def __init__(self, config: MUDefenseEvaluationConfig,**kwargs):
        """Initialize the evaluator."""
        super().__init__(config, **kwargs)
        self.config = config.__dict__
        for key, value in kwargs.items():
            setattr(config, key, value)
        self.job = self.config.get("job") 
        self.gen_imgs_path = self.config.get("gen_imgs_path")
        self.coco_imgs_path = self.config.get("coco_imgs_path")
        self.prompt_path = self.config.get("prompt_path")
        self.classify_prompt_path = self.config.get("classify_prompt_path")
        self.classification_model_path = self.config.get("classification_model_path")
        self.devices = self.config.get("devices")
        self.output_path = self.config.get("output_path")
        self.devices = [f'cuda:{int(d.strip())}' for d in self.devices.split(',')]

        self.logger = logging.getLogger(__name__)

        self._parse_config()
        self.load_model()
        config.validate_config()
        self.results = {}

    
    def load_model(self):
        """Load models needed for evaluation."""
        if self.job == 'clip':
            self.clip_model = CLIPModel.from_pretrained(self.classification_model_path).to(self.devices[0])
            self.clip_processor = CLIPProcessor.from_pretrained(self.classification_model_path)
        else:
            self.clip_model = None
            self.clip_processor = None
    
    def calculate_clip_score(self):
        """Calculate the mean CLIP score over generated images using prompts."""
        df = pd.read_csv(self.prompt_path)
        clip_scores = []
        
        for count, (_, row) in enumerate(df.iterrows(), start=1):
            case_num = row['case_number']
            # Construct the image path; assumes images are named like '{case_number}_0.png'
            img_path = os.path.join(self.gen_imgs_path, f'{case_num}_0.png')
            try:
                image = Image.open(img_path)
            except Exception as e:
                self.logger.error(f"Error loading image {img_path}: {e}")
                continue
            text = row['prompt']
            
            # Prepare inputs for CLIP
            inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.devices[0]) for k, v in inputs.items()}
            
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_scores.append(logits_per_image.item())
            
            if count % 100 == 0:
                self.logger.info(f"Processed {count} images")
        
        if clip_scores:
            average_clip_score = sum(clip_scores) / len(clip_scores)
        else:
            average_clip_score = 0.0
        
        result_str = f"{average_clip_score}"
        self.results['clip score'] = result_str
        return result_str
    
    def calculate_fid_score(self):
        """Calculate the Fréchet Inception Distance (FID) score."""

        fid, _ = calculate_fid(self.gen_imgs_path, self.coco_imgs_path)
        result_str = f"{fid}"
        self.results['fid'] = result_str
        return result_str

    def save_results(self, result_data):
        """Save the evaluation results to a JSON file."""
        # Choose file name based on the results available.
        if "fid" in result_data and "clip" in result_data:
            file_path = self.output_path + '_results.json'
        elif "fid" in result_data:
            file_path = self.output_path + '_fid.json'
        elif "clip" in result_data:
            file_path = self.output_path + '_clip.json'
        else:
            file_path = self.output_path + '_results.json'
        
        # Create the output directory if it does not exist
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir) and output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(result_data, file, indent=4)
        
        self.logger.info(f"Results saved to {file_path}")


    def run(self):
        """Run the evaluation process."""
        # If no job type is mentioned (i.e. self.job is None or empty),
        # calculate both FID and CLIP scores.
        if not self.job:
            fid_result = self.calculate_fid_score()
            clip_result = self.calculate_clip_score()
            result_data = {
                "fid": fid_result,
                "clip": clip_result
            }
        elif self.job == 'fid':
            fid_result = self.calculate_fid_score()
            result_data = {"fid": fid_result}
        elif self.job == 'clip':
            clip_result = self.calculate_clip_score()
            result_data = {"clip": clip_result}
        else:
            raise ValueError(f"Unsupported job type: {self.job}")

        self.logger.info(result_data)
        self.save_results(result_data)


