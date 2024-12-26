import os
import argparse
import logging
from mu.algorithms.unified_concept_editing.algorithm import (
    UnifiedConceptEditingAlgorithm,
)
from mu.helpers.path_setup import *
from mu.helpers import load_config, setup_logger

def main():
    parser = argparse.ArgumentParser(
        prog="TrainUSD",
        description="Finetuning Stable Diffusion model to debias the concepts",
    )

    # Original Arguments from the first main function
    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
    )

    # New Arguments from the second main function
    parser.add_argument(
        "--concept",
        help="Prompt corresponding to concept to erase",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--attributes", help="Attributes to debias", type=str, default="male, female"
    )
    parser.add_argument(
        "--device", help="CUDA device to train on", type=str, default="0"
    )
    parser.add_argument(
        "--base", help="Base version for Stable Diffusion", type=str, default="1.4"
    )
    parser.add_argument(
        "--num_images",
        help="Number of images per concept to generate for validation",
        type=int,
        default=10,
    )

    # Training parameters
    parser.add_argument(
        "--train_method", type=str, choices=["full", "partial"], help="Training method"
    )
    parser.add_argument("--alpha", type=float, help="Alpha parameter")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--dataset_type", type=str, choices=["unlearncanvas", "i2p"])
    parser.add_argument("--template", type=str, choices=["object", "style", "i2p"])
    parser.add_argument(
        "--template_name", type=str, choices=["self-harm", "Abstractionism"]
    )
    parser.add_argument(
        "--devices", help="CUDA devices to train on (comma-separated)", type=str
    )
    parser.add_argument("--use_sample", type=bool, help="Whether to use sampling")
    parser.add_argument(
        "--guided_concepts", type=str, help="Concepts to guide the editing"
    )
    parser.add_argument(
        "--technique", type=str, choices=["replace", "tensor"], help="Editing technique"
    )
    parser.add_argument("--preserve_scale", type=float, help="Scale for preservation")
    parser.add_argument("--preserve_number", type=int, help="Number to preserve")
    parser.add_argument("--erase_scale", type=float, help="Scale for erasure")
    parser.add_argument("--lamb", type=float, help="Lambda parameter")
    parser.add_argument("--add_prompts", type=bool, help="Whether to add prompts")

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

    config["devices"] = devices

    # Setup logger
    log_file = os.path.join(
        logs_dir,
        f"uce_training_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Unified Concept Editing(Debias) Training")

    # Initialize and run the UnifiedConceptEditingAlgorithm
    algorithm = UnifiedConceptEditingAlgorithm(config)
    algorithm.run(erase="debias")


if __name__ == "__main__":
    main()
