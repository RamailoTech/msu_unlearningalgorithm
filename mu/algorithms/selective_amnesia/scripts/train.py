# mu/algorithms/selective_amnesia/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm
from mu.helpers import setup_logger, load_config, str2bool
from mu.helpers.path_setup import logs_dir
from mu.algorithms.selective_amnesia.configs import SelectiveAmnesiaConfig


def main():
    parser = argparse.ArgumentParser(description="Train Selective Amnesia")

    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--full_fisher_dict_pkl_path",
        help="Checkpoint path for Stable Diffusion",
        type=str,
        required=True,
    )

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

    # Device configuration
    parser.add_argument(
        "--devices", help="CUDA devices to train on (comma-separated)", type=str
    )

    # Additional flags
    parser.add_argument("--use_sample", help="Use the sample dataset for training")

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)

    # Prepare output directory
    os.makedirs(args.output_dir or config.get("output_dir", "results"), exist_ok=True)

    # Update configuration only if arguments are explicitly provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    # Setup logger
    log_file = os.path.join(
        logs_dir,
        f"selective_amnesia_training_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Selective amnesia Training")

    # Initialize and run the EraseDiff algorithm
    config = SelectiveAmnesiaConfig(**config)
    algorithm = SelectiveAmnesiaAlgorithm(config, config_path=args.config_path)
    algorithm.run()


if __name__ == "__main__":
    main()
