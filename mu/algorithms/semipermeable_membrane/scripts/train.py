# semipermeable_membrane/scripts/train.py

import argparse
import logging
import os

import yaml
from algorithms.semipermeable_membrane.algorithm import SemipermeableMembraneAlgorithm
from algorithms.semipermeable_membrane.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train Semipermeable Membrane Algorithm"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup logger
    output_dir = config.get("save", {}).get("path", "checkpoints/")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "spm_training.log")
    logger = setup_logger(name="SPMTraining", log_file=log_file, level=logging.INFO)
    logger.info("Starting Semipermeable Membrane Training")

    # Initialize and run the SemipermeableMembraneAlgorithm
    algorithm = SemipermeableMembraneAlgorithm(config)
    algorithm.run()


if __name__ == "__main__":
    main()
