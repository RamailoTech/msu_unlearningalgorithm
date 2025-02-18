# mu_attack/execs/attack.py

import os
import sys
import json
import argparse
import random
import torch
import numpy as np
from datetime import datetime
import logging
from pydantic import ValidationError,BaseModel

from importlib import import_module


class MUAttack:
    """
    Main orchestration class that:
      1) Loads a config (from file or dict),
      2) Sets random seeds,
      3) Initializes the Task,
      4) Initializes the Attacker,
      5) Initializes the Logger,
      6) Runs the attack.
    """

    def __init__(self, config,  quiet=False,**overridable_params):
        """
        :param config_path: Path to a JSON config file (optional).
        :param config_dict: A dict containing the config (optional).
        :param quiet: Whether to suppress printing config summary.
        """
        self.logger = logging.getLogger(__name__)
        self.config = None

        self.config = config.model_dump()
        self._update_config_from_kwargs(self.config, overridable_params)
        self.validate_config_json()

        self.load_resume_config()
        self.validate_config()
        if not quiet:
            self.print_config()
        self.setup_seed()
        self.init_task()
        self.init_attacker()
        self.init_logger()
        self.run()

    

    def _update_config_from_kwargs(self, config_dict, overrides):
        """
        Updates config dictionary using dot-separated keys.
        Supports both Pydantic models and dictionary fields.
        """
        for key, value in overrides.items():
            keys = key.split(".")  # Convert 'task.name' -> ['task', 'name']
            current_dict = config_dict

            for attr in keys[:-1]:  # Traverse to the final attribute
                if attr not in current_dict:
                    raise AttributeError(f"Invalid config key: {key} (Could not find '{attr}')")
                current_dict = current_dict[attr]

            final_attr = keys[-1]
            if final_attr in current_dict:
                current_dict[final_attr] = value  # Override value
            else:
                raise AttributeError(f"Invalid config key: {key} (Could not find '{final_attr}')")

    
    def validate_config_json(self):
        """
        Validates the configuration, ensuring all required file paths exist.
        """
        task_config = self.config.get("task", {})
        backend = task_config.get("backend")
        
        compvis_ckpt_path = task_config.get("compvis_ckpt_path")
        compvis_config_path = task_config.get("compvis_config_path")
        dataset_path = task_config.get("dataset_path")
        diffusers_model_name_or_path = task_config.get("diffusers_model_name_or_path")
        target_ckpt = task_config.get("target_ckpt")

        if backend == "compvis":
            if compvis_ckpt_path and not os.path.exists(compvis_ckpt_path):
                raise FileNotFoundError(f"Checkpoint path does not exist: {compvis_ckpt_path}")
            
            if compvis_config_path and not os.path.exists(compvis_config_path):
                raise FileNotFoundError(f"Config path does not exist: {compvis_config_path}")

        #Validate diffusers paths only if backend is "diffusers"
        if backend == "diffusers":
            if diffusers_model_name_or_path and not os.path.exists(diffusers_model_name_or_path):
                raise FileNotFoundError(f"Diffusers model path does not exist: {diffusers_model_name_or_path}")

            # if target_ckpt and not os.path.exists(target_ckpt):
            #     raise FileNotFoundError(f"Target checkpoint does not exist: {target_ckpt}")

        # Always validate dataset path (common to both backends)
        if dataset_path and not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    def load_resume_config(self):
        """
        If 'overall.resume' is present, load that config.json and merge it
        into self.config.
        """
        resume_path = self.config.get("overall", {}).get("resume", None)
        if resume_path is not None:
            resume_file = os.path.join(resume_path, "config.json")
            if os.path.isfile(resume_file):
                with open(resume_file, "r") as f:
                    resume_config = json.load(f)
                self.merge_dicts(self.config, resume_config)
                self.logger.info(f"Successfully merged config from {resume_file}")
            else:
                self.logger.warning(
                    f"No config.json found at {resume_file}, skipping resume."
                )

    def merge_dicts(self, base, override):
        """
        Recursively merge 'override' into 'base'.
        Values in 'override' take precedence.
        """
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self.merge_dicts(base[k], v)
            else:
                base[k] = v

    def validate_config(self):
        """
        Perform manual checks on the config.
        """
        overall = self.config.get("overall", {})
        if "task" not in overall:
            raise ValueError("Config validation error: 'overall.task' is required.")
        if "attacker" not in overall:
            raise ValueError("Config validation error: 'overall.attacker' is required.")
        if "logger" not in overall:
            raise ValueError("Config validation error: 'overall.logger' is required.")

        # Example: Check that the dataset_path is a valid directory
        task_cfg = self.config.get("task", {})
        if not os.path.isdir(task_cfg.get("dataset_path", "")):
            raise ValueError(
                "Config validation error: `task.dataset_path` must be a valid directory."
            )

    def print_config(self):
        self.logger.info("===== CONFIG =====")
        self.logger.info(json.dumps(self.config, default=str,indent=4))
        self.logger.info("==================\n")

    def setup_seed(self):
        """
        Sets random seeds for reproducibility.
        """
        # If 'seed' not given, default to 0
        seed = self.config.get("overall", {}).get("seed", 0)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # For deterministic behavior
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_task(self):
        """
        Dynamically import and create the Task instance
        (e.g., classifier, sd_guidence, P4D, etc.).
        """
        task_name = self.config["overall"]["task"]  # e.g., "P4D"
        # Base task section
        task_config = self.config.get("task", {})
        if task_config.get(task_name):
            task_config = {**task_config, **task_config[task_name]}
            del task_config[task_name]
        module_path = f"mu_attack.tasks.{task_name.lower()}"
        task_module = import_module(module_path)

        self.task = task_module.get(**task_config)

    def init_attacker(self):
        """
        Dynamically import and create the Attacker instance
        (e.g., gcg, text_grad, hard_prompt, no_attack, etc.).
        """
        attacker_name = self.config["overall"]["attacker"]  # e.g. "hard_prompt"
        # Merged attacker config
        attacker_config = self.config.get("attacker", {})

        # Some attackers have a sub-section with the same name,
        sub_config = attacker_config.get(attacker_name, {})
        if attacker_config.get(attacker_name):
            del attacker_config[attacker_name]
        # Merge them so that, for example, "lr", "weight_decay" get included:
        final_config = {**attacker_config, **sub_config}

        # Dynamically import the correct module:
        module_path = f"mu_attack.attackers.{attacker_name}"
        attacker_module = import_module(module_path)
        self.attacker = attacker_module.get(**final_config)

    def init_logger(self):
        """
        Dynamically import and create the Logger instance
        (e.g., json or none).
        """
        logger_name = self.config["overall"]["logger"]  # e.g. "json"
        logger_config = self.config.get("logger", {})
        # Similarly, check for sub-config:
        sub_config = logger_config.get(logger_name, {})
        del logger_config[logger_name]
        final_config = {**logger_config, **sub_config}

        # Add the entire config if the logger needs it
        final_config["config"] = self.config

        module_path = f"mu_attack.loggers.{logger_name}_"
        logger_module = import_module(module_path)

        self.logger = logger_module.get(**final_config)

    def run(self):
        """
        Kick off the actual attack:
         attacker.run(task, logger).
        """
        self.attacker.run(self.task, self.logger)


def parse_args():
    """
    Simple argument parser if you also want to allow:
      python attack.py --config_path /path/to/config.json
    """
    parser = argparse.ArgumentParser("Diffusion MU Attack (Refactored)")
    parser.add_argument(
        "--config_path", type=str, help="Path to the JSON config file", required=True
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False, help="Suppress config printing"
    )
    parser.add_argument(
        "--attack_idx",
        type=int,
        default=None,
        help="Override attacker.attack_idx from the JSON config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_dict = (
        {"attacker": {"attack_idx": args.attack_idx}} if args.attack_idx else None
    )
    main = MUAttack(config_path=args.config_path, quiet=args.quiet)
