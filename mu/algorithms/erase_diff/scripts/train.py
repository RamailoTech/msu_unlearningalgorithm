# erase_diff/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from algorithms.erase_diff.algorithm import EraseDiffAlgorithm
from mu.helpers.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        prog="TrainEraseDiff",
        description="Finetuning Stable Diffusion model to erase concepts using the EraseDiff method",
    )

    # Training parameters
    parser.add_argument(
        "--train_method",
        help="method of training",
        type=str,
        default="xattn",
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
        "--alpha",
        help="Guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs to train",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument("--K_steps", type=int, required=False, default=2)
    parser.add_argument(
        "--lr",
        help="Learning rate used to train",
        type=float,
        required=False,
        default=5e-5,
    )

    # Model configuration
    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=False,
        default="configs/train_erase_diff.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="Checkpoint path for Stable Diffusion",
        type=str,
        required=False,
        default="path/to/checkpoint.ckpt",
    )

    # Dataset directories
    parser.add_argument(
        "--original_data_dir",
        type=str,
        required=False,
        default="/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/quick-canvas-benchmark",
        help="Directory containing the original dataset organized by themes and classes.",
    )
    parser.add_argument(
        "--new_data_dir",
        type=str,
        required=False,
        default="/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/erase_diff/data",
        help="Directory where the new datasets will be saved.",
    )

    # Output configurations
    parser.add_argument(
        "--output_dir",
        help="Output directory to save results",
        type=str,
        required=False,
        default="results",
    )
    parser.add_argument(
        "--theme", type=str, required=True, help="Concept or theme to unlearn"
    )
    parser.add_argument(
        "--classes", type=str, required=True, help="Class or objects to unlearn"
    )

    parser.add_argument(
        "--separator",
        help="Separator if you want to train multiple words separately",
        type=str,
        required=False,
        default=None,
    )

    # Sampling and image configurations
    parser.add_argument(
        "--image_size",
        help="Image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--interpolation",
        help="Interpolation mode",
        type=str,
        required=False,
        default="bicubic",
        choices=["bilinear", "bicubic", "lanczos"],
    )
    parser.add_argument(
        "--ddim_steps",
        help="DDIM steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--ddim_eta", help="DDIM eta parameter", type=float, required=False, default=0.0
    )

    # Device configuration
    parser.add_argument(
        "--devices",
        help="CUDA devices to train on (comma-separated)",
        type=str,
        required=False,
        default="0",
    )

    # Additional flags
    parser.add_argument(
        "--use_sample", action="store_true", help="Use the sample dataset for training"
    )
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    parser.add_argument("--pin_memory", type=bool, required=False, default=True)

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join(args.output_dir, "erase_diff_training.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting EraseDiff Training")

    # Prepare output directory
    output_name = os.path.join(args.output_dir, f"{args.theme}.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving the model to {output_name}")

    # Parse devices
    devices = [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]
    # Create configuration dictionary
    config = {
        "train_method": args.train_method,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "K_steps": args.K_steps,
        "lr": args.lr,
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "original_data_dir": args.original_data_dir,
        "new_data_dir": args.new_data_dir,
        "output_dir": args.output_dir,
        "theme": args.theme,
        "class": args.classes,
        "separator": args.separator,
        "image_size": args.image_size,
        "interpolation": args.interpolation,
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "devices": devices,
        "output_name": output_name,
        "use_sample": args.use_sample,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    # Initialize and run the EraseDiff algorithm
    algorithm = EraseDiffAlgorithm(config)
    algorithm.run()


if __name__ == "__main__":
    main()
