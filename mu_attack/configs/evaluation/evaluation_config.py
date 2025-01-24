# mu_attack/configs/evaluation/evaluation_config.py

import os
from pathlib import Path

from mu.core.base_config import BaseConfig

current_dir = Path(__file__).parent


class ASRConfig(BaseConfig):
    
    def __init__(self,
                 root,
                  root_no_attack):
        self.root = root
        self.root_no_attack = root_no_attack

class ClipConfig(BaseConfig):
    def __init__(self,
                 image_path,
                 log_path,
                 model_name_or_path="openai/clip-vit-base-patch32",
                 devices = "0"
                 ):
        self.image_path = image_path
        self.log_path = log_path
        self.model_name_or_path = model_name_or_path
        self.devices = devices

class FidConfig(BaseConfig):
    def __init__(self,
                 ref_batch_path,
                 sample_batch_path):
        self.ref_batch_path = ref_batch_path
        self.sample_batch_path = sample_batch_path


class AttackEvaluatorConfig(BaseConfig):

    def __init__(self,
                 asr_root="results/random_esd_nudity/Hard Prompt",  # Default path for attack results
                 asr_root_no_attack="results/no_attack_esd_nudity/NoAttackEsdNudity",  # Default path for no attack results
                 image_path="results/random_esd_nudity/Hard Prompt/images",  # Default path for images to calculate clip score
                 log_path="results/random_esd_nudity/Hard Prompt/log.json",  # path for logs to extract prompts to calculate clip score.
                 model_name_or_path="openai/clip-vit-base-patch32",  # Default model name
                 devices="0",  # Default device
                 fid_ref_batch_path="results/hard_prompt_esd_nudity_P4D/P4d/images",  # Default path for reference batch
                 fid_sample_batch_path="outputs/dataset/i2p_nude/imgs",  # Default path for sample batch
                 output_path="results/evaluation/results.json",  # Default output path
                 **kwargs):
        # Initialize ASRConfig with the provided or default paths
        self.asr = ASRConfig(
            root=asr_root,
            root_no_attack=asr_root_no_attack
        )

        # Initialize ClipConfig with the provided or default paths and parameters
        self.clip = ClipConfig(
            image_path=image_path,
            log_path=log_path,
            model_name_or_path=model_name_or_path,
            devices=devices
        )

        # Initialize FidConfig with the provided or default paths
        self.fid = FidConfig(
            ref_batch_path=fid_ref_batch_path,
            sample_batch_path=fid_sample_batch_path
        )

        # Set the output path
        self.output_path = output_path

        # Process additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)


    def validate_config(self):
        """
        Perform basic validation on the config parameters.
        """

        if not os.path.exists(self.asr.root):
            raise FileNotFoundError(f"Result directory {self.asr.root} does not exist.")
        if not os.path.exists(self.asr.root_no_attack):
            raise FileNotFoundError(f"Result directory {self.asr.root_no_attack} does not exist.")
        if not os.path.exists(self.clip.image_path):
            raise FileNotFoundError(f"Result directory {self.clip.image_path} does not exist.")
        if not os.path.exists(self.fid.ref_batch_path):
            raise FileNotFoundError(f"Result directory {self.fid.ref_batch_path} does not exist.")
        if not os.path.exists(self.fid.sample_batch_path):
            raise FileNotFoundError(f"Dataset directory {self.fid.sample_batch_path} does not exist.")


attack_evaluation_config = AttackEvaluatorConfig()
