# mu_defense/algorithms/adv_unlearn/evaluator.py

import os
import pandas as pd
import json
import logging

from mu_defense.algorithms.adv_unlearn.configs import MUDefenseEvaluationConfig

# from evaluation.metrics.clip import ClipScoreEvaluator
from evaluation.core import BaseEvaluator
# from evaluation.metrics.fid import calculate_fid_score
# from mu_defense.algorithms.adv_unlearn.configs import example_image_generator_config
from mu_defense.algorithms.adv_unlearn.generate_example_image import ImageGenerator

class MUDefenseEvaluator(BaseEvaluator):
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
        self.prompt_path = self.config.get("prompt_file_path")
        self.classify_prompt_path = self.config.get("classify_prompt_path")
        self.classification_model_path = self.config.get("classification_model_path")
        self.devices = self.config.get("devices")
        self.output_path = self.config.get("output_path")
        self.devices = [f'cuda:{int(d.strip())}' for d in self.devices.split(',')]
        self.image_generator = None
        self.logger = logging.getLogger(__name__)

        self._parse_config()
        # self.load_model()
        config.validate_config()
        self.results = {}

    def sampler(self):
        self.image_generator = ImageGenerator(self.config)

    def generate_images(self):
        self.sampler()
        gen_img_path = self.image_generator.generate_images()
        return gen_img_path

    
    # def calculate_clip_score(self):
    #     """Calculate the mean CLIP score over generated images using prompts."""

    #     clip_evaluator = ClipScoreEvaluator(
    #         gen_image_path = self.gen_imgs_path,
    #         prompt_file_path = self.prompt_path,
    #         devices = self.devices,
    #         classification_model_path = self.classification_model_path
    #     )
    #     res = clip_evaluator.compute_clip_score()
    #     return res

    
    # def calculate_fid_score(self):
    #     """Calculate the Fréchet Inception Distance (FID) score."""

    #     fid, _ = calculate_fid_score(self.gen_imgs_path, self.coco_imgs_path)
    #     result_str = f"{fid}"
    #     self.results['fid'] = result_str
    #     return result_str

    # def save_results(self, result_data):
    #     """Save the evaluation results to a JSON file."""
    #     # Choose file name based on the results available.
    #     if "fid" in result_data and "clip" in result_data:
    #         file_path = self.output_path + '_results.json'
    #     elif "fid" in result_data:
    #         file_path = self.output_path + '_fid.json'
    #     elif "clip" in result_data:
    #         file_path = self.output_path + '_clip.json'
    #     else:
    #         file_path = self.output_path + '_results.json'
        
    #     # Create the output directory if it does not exist
    #     output_dir = os.path.dirname(file_path)
    #     if not os.path.exists(output_dir) and output_dir != "":
    #         os.makedirs(output_dir, exist_ok=True)
        
    #     with open(file_path, 'w', encoding='utf-8') as file:
    #         json.dump(result_data, file, indent=4)
        
    #     self.logger.info(f"Results saved to {file_path}")


    # def run(self):
    #     """Run the evaluation process."""
    #     # If no job type is mentioned (i.e. self.job is None or empty),
    #     # calculate both FID and CLIP scores.
    #     if not self.job:
    #         fid_result = self.calculate_fid_score()
    #         clip_result = self.calculate_clip_score()
    #         result_data = {
    #             "fid": fid_result,
    #             "clip": clip_result
    #         }
    #     elif self.job == 'fid':
    #         fid_result = self.calculate_fid_score()
    #         result_data = {"fid": fid_result}
    #     elif self.job == 'clip':
    #         clip_result = self.calculate_clip_score()
    #         result_data = {"clip": clip_result}
    #     else:
    #         raise ValueError(f"Unsupported job type: {self.job}")

    #     self.logger.info(result_data)
    #     self.save_results(result_data)


