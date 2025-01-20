# mu/algorithms/esd/scripts/train.py

import argparse
import os
import logging

from mu.algorithms.esd.algorithm import ESDAlgorithm
from mu.helpers import setup_logger, load_config, setup_logger
from mu.helpers.path_setup import *
from mu.algorithms.esd.configs import ESDConfig


def main():
    parser = argparse.ArgumentParser(
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
    )

    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
    )
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
        "--start_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--negative_guidance",
        help="guidance of negative training used to train",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--iterations", help="iterations used to train", type=int, required=False
    )
    parser.add_argument(
        "--lr", help="learning rate used to train", type=float, required=False
    )
    parser.add_argument(
        "--image_size", help="image size used to train", type=int, required=False
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
    )

    # Model configuration
    parser.add_argument(
        "--model_config_path", help="Model Config path for Stable Diffusion", type=str
    )
    parser.add_argument(
        "--ckpt_path", help="Checkpoint path for Stable Diffusion", type=str
    )

    # Dataset directories
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

    # Device configuration
    parser.add_argument(
        "--devices", help="CUDA devices to train on (comma-separated)", type=str
    )
    parser.add_argument("--use_sample", help="Use the sample dataset for training")

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
        f"erase_diff_training_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting EraseDiff Training")

    # Initialize and run the EraseDiff algorithm
    config = ESDConfig(**config)
    algorithm = ESDAlgorithm(config)
    algorithm.run()


if __name__ == "__main__":
    main()
