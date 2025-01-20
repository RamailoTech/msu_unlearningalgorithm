# mu/algorithms/saliency_unlearning/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import *
from mu.algorithms.saliency_unlearning.configs import SaliencyUnlearningConfig


def main():
    parser = argparse.ArgumentParser(
        prog="SaliencyUnlearnTrain",
        description="Finetuning stable diffusion model to perform saliency-based unlearning.",
    )

    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--alpha",
        help="Guidance scale for loss combination",
        type=float,
    )
    parser.add_argument("--epochs", help="Number of epochs to train", type=int)
    parser.add_argument(
        "--train_method",
        help="Method of training",
        type=str,
        choices=[
            "noxattn",
            "selfattn",
            "xattn",
            "full",
            "notime",
            "xlayer",
            "selflayer",
        ],
    )
    parser.add_argument(
        "--model_config_path", type=str, help="Path to the model configuration file"
    )

    # Dataset directories
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        help="Directory containing the original dataset organized by themes and classes.",
    )
    parser.add_argument(
        "--processed_dataset_dir",
        type=str,
        help="Directory where the new datasets will be saved.",
    )
    parser.add_argument("--dataset_type", type=str, choices=["unlearncanvas", "i2p"])
    parser.add_argument("--template", type=str, choices=["object", "style", "i2p"])
    parser.add_argument(
        "--template_name", type=str, choices=["self-harm", "Abstractionism"]
    )

    parser.add_argument(
        "--output_dir", help="Output directory for the masks ", type=str
    )
    parser.add_argument("--mask_path", help="Path to the mask file", type=str)
    parser.add_argument("--use_sample", help="Use the sample dataset for training")

    parser.add_argument("--start_guidance", help="Starting guidance factor", type=float)
    parser.add_argument(
        "--negative_guidance", help="Negative guidance factor", type=float
    )
    parser.add_argument(
        "--ddim_steps", help="DDIM steps of inference used to train", type=int
    )

    # Device configuration
    parser.add_argument(
        "--devices", help="CUDA devices to train on (comma-separated)", type=str
    )

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)

    # Prepare output directory
    output_name = os.path.join(
        args.output_dir or config.get("output_dir", "results"),
        f"{args.template_name or config.get('template_name', 'self-harm')}.pth",
    )
    os.makedirs(args.output_dir or config.get("output_dir", "results"), exist_ok=True)

    # Update configuration only if arguments are explicitly provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    # Setup logger
    log_file = os.path.join(
        logs_dir,
        f"saliency_unlearning_masking_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Saliency Unlearning Masking")

    config = SaliencyUnlearningConfig(**config)
    algorithm = SaliencyUnlearnAlgorithm(config)
    algorithm.run()


if __name__ == "__main__":
    main()
