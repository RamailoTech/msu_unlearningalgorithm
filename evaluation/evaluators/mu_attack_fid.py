# mu_attack/evaluators/fid.py

from typing import Any, Dict
import os
import json
import numpy as np
from PIL import Image
import logging

import tensorflow.compat.v1 as tf

from mu.core.base_config import BaseConfig
from mu_attack.configs.evaluation import AttackEvaluatorConfig
from evaluation.core import AttackBaseEvaluator
from mu_attack.tasks.utils.ldm.modules.evaluate.adm_evaluator import Evaluator


class FIDEvaluator(AttackBaseEvaluator):
    """
    FIDEvaluator for computing FID, Inception Score, and other evaluation metrics.
    """

    def __init__(self, config:AttackEvaluatorConfig, **kwargs):
        """
        Initialize the FID Evaluator with batch paths and TensorFlow evaluator setup.

        Args:
            ref_batch_path (str): Path to the reference batch npz file.
            sample_batch_path (str): Path to the sample batch npz file.
        """

        super().__init__(config, **kwargs)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        for key, value in kwargs.items():
            if not hasattr(config, key):
                setattr(config, key, value)
                continue
            config_attr = getattr(config, key)
            if isinstance(config_attr, BaseConfig) and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    setattr(config_attr, sub_key, sub_val)
            elif isinstance(config_attr, dict) and isinstance(value, dict):
                config_attr.update(value)
            else:
                setattr(config, key, value)

        self.config = config.to_dict()
        config.validate_config()
        self.output_path = self.config.get('output_path')
        self.config = self.config.get("fid", {})
        self.ref_batch_path = self.config['ref_batch_path']
        self.sample_batch_path = self.config['sample_batch_path']        

        self.result = {}

        # Configure TensorFlow for evaluation
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.evaluator = Evaluator(tf.Session(config=tf_config))

        #paths for npz file
        self.npz_ref_path = f"{self.ref_batch_path}/image_dataset.npz"
        self.npz_sample_path = f"{self.sample_batch_path}/image_dataset.npz"

        self.logger = logging.getLogger(__name__)

    def load_and_prepare_data(self):
        """
        Load and prepare data by warming up the evaluator and computing activations.
        """
        self.logger.info("Warming up TensorFlow...")
        self.evaluator.warmup()   

        self.logger.info("Computing reference batch activations...")
        self.ref_acts = self.evaluator.read_activations(self.npz_ref_path)

        self.logger.info("Computing/reading reference batch statistics...")
        self.ref_stats, _ = self.evaluator.read_statistics(self.npz_ref_path, self.ref_acts)

        self.logger.info("Computing sample batch activations...")
        self.sample_acts = self.evaluator.read_activations(self.npz_sample_path)

        self.logger.info("Computing/reading sample batch statistics...")
        self.sample_stats, _ = self.evaluator.read_statistics(self.npz_sample_path, self.sample_acts)

    def compute_score(self):
        """
        Compute the FID, Inception Score, sFID, Precision, and Recall metrics.
        """
        self.logger.info("Computing evaluation metrics...")

        # FID and Spatial FID
        self.result['FID'] = self.sample_stats.frechet_distance(self.ref_stats)
        self.logger.info(f"FID: {self.result['FID']}")

    def images_to_npz(self,folder_path,npz_path,target_size=(224, 224)):
        """
        Convert all images in a folder to a .npz file.

        Args:
            folder_path (str): Path to the folder containing images.
            output_path (str): Path to save the .npz file.
            target_size (tuple): Size to resize images (width, height). Default is (224, 224).
        """
        image_data = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                with Image.open(file_path) as img:
                    img = img.convert("RGB").resize(target_size)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    image_data.append(img_array)
        
        # Save all images as a single key "arr_0"
        np.savez(npz_path, arr_0=np.array(image_data))
        self.logger.info(f"Saved to {npz_path} with key 'arr_0'.")


    def save_results(self):
        """
        Save or append the CLIP score results to a JSON file.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        existing_data = {}

        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            except json.JSONDecodeError:
                pass  # Ignore if the file is invalid

        if isinstance(existing_data, dict):
            existing_data.update(self.result)
        else:
            self.logger.warning(f"Unexpected JSON format in {self.output_path}. Overwriting file.")
            existing_data = self.result

        # Write the updated data back to the file
        with open(self.output_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        self.logger.info(f'Results saved to {self.output_path}')

    def run(self):
        """
        Run the FID evaluator.
        """
        self.logger.info("Calculating FID score...")

        self.logger.info("Converting reference images to .npz...")
        self.images_to_npz(self.ref_batch_path, self.npz_ref_path)

        self.logger.info("Converting sample images to .npz...")
        self.images_to_npz(self.sample_batch_path, self.npz_sample_path)

        # Load and prepare data
        self.load_and_prepare_data()

        # Compute fid score
        self.compute_score()

        # Save results
        self.save_results()


 




