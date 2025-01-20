# mu/algorithms/erase_diff/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.algorithms.erase_diff.configs.train_config import (
    EraseDiffConfig,
    erase_diff_train_config_quick_canvas,
)
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import logs_dir


def main():
    parser = argparse.ArgumentParser(
        prog="TrainEraseDiff",
        description="Finetuning Stable Diffusion model to erase concepts using the EraseDiff method",
    )

    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
    )

    # Training parameters
    parser.add_argument(
        "--train_method",
        help="method of training",
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
        "--alpha", help="Guidance of start image used to train", type=float
    )
    parser.add_argument("--epochs", help="Number of epochs to train", type=int)
    parser.add_argument("--K_steps", type=int)
    parser.add_argument("--lr", help="Learning rate used to train", type=float)

    # Model configuration
    parser.add_argument(
        "--model_config_path", help="Model Config path for Stable Diffusion", type=str
    )
    parser.add_argument(
        "--ckpt_path", help="Checkpoint path for Stable Diffusion", type=str
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

    # Output configurations
    parser.add_argument(
        "--output_dir", help="Output directory to save results", type=str
    )
    parser.add_argument(
        "--separator",
        help="Separator if you want to train multiple words separately",
        type=str,
    )

    # Sampling and image configurations
    parser.add_argument("--image_size", help="Image size used to train", type=int)
    parser.add_argument(
        "--interpolation",
        help="Interpolation mode",
        type=str,
        choices=["bilinear", "bicubic", "lanczos"],
    )
    parser.add_argument(
        "--ddim_steps", help="DDIM steps of inference used to train", type=int
    )
    parser.add_argument("--ddim_eta", help="DDIM eta parameter", type=float)

    # Device configuration
    parser.add_argument(
        "--devices", help="CUDA devices to train on (comma-separated)", type=str
    )

    # Additional flags
    parser.add_argument("--use_sample", help="Use the sample dataset for training")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--pin_memory", type=bool)

    # Newly added parameters
    parser.add_argument(
        "--start_guidence", help="(newly added) Starting guidance factor", type=float
    )
    parser.add_argument(
        "--negative_guidance", help="(newly added) Negative guidance factor", type=float
    )

    args = parser.parse_args()
    config = erase_diff_train_config_quick_canvas

    # Prepare output directory
    output_name = os.path.join(
        args.output_dir or config.output_dir or "results",
        f"{args.template_name or config.template_name or  'self-harm'}.pth",
    )
    os.makedirs(args.output_dir or config.output_dir or "results", exist_ok=True)

    # Update configuration only if arguments are explicitly provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    # Setup logger
    log_file = os.path.join(
        logs_dir,
        f"erase_diff_training_{config.dataset_type}_{config.template}_{config.template_name}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting EraseDiff Training")

    algorithm = EraseDiffAlgorithm(config)
    algorithm.run()


if __name__ == "__main__":
    main()
