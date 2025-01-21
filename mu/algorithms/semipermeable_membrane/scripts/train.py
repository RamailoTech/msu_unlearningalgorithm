# mu/algorithms/semipermeable_membrane/scripts/train.py

import argparse
import os
import yaml
import logging

from mu.algorithms.semipermeable_membrane import SemipermeableMembraneAlgorithm
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import logs_dir
from mu.algorithms.semipermeable_membrane.configs import SemipermeableMembraneConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train Semipermeable Membrane Algorithm"
    )
    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
    )

    parser.add_argument("--dataset_type", type=str, choices=["unlearncanvas", "i2p"])
    parser.add_argument("--template", type=str, choices=["object", "style", "i2p"])
    parser.add_argument(
        "--template_name", type=str, choices=["self-harm", "Abstractionism"]
    )
    parser.add_argument(
        "--devices", help="CUDA devices to train on (comma-separated)", type=str
    )

    parser.add_argument(
        "--output_dir", help="Output directory to save results", type=str
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

    # Parse devices
    devices = (
        [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]
        if args.devices
        else [f"cuda:{int(d.strip())}" for d in config.get("devices").split(",")]
    )

    # Update configuration only if arguments are explicitly provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    # Ensure devices are properly set
    config["devices"] = devices

    # Setup logger
    log_file = os.path.join(
        logs_dir,
        f"erase_diff_training_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting EraseDiff Training")

    # Initialize and run the SemipermeableMembraneAlgorithm
    config = SemipermeableMembraneConfig(**config)
    algorithm = SemipermeableMembraneAlgorithm(config)
    algorithm.run()


if __name__ == "__main__":
    main()
