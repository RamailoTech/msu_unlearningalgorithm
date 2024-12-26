# saliency_unlearning/scripts/train.py

import argparse
import logging
import os
import sys

import torch
from algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
from algorithms.saliency_unlearning.logger import setup_logger
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SaliencyUnlearnTrain",
        description="Finetuning stable diffusion model to perform saliency-based unlearning.",
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

    parser.add_argument(
        "--theme", type=str, required=True, help="Concept or theme to unlearn"
    )
    parser.add_argument(
        "--classes", type=str, required=True, help="Class or objects to unlearn"
    )
    parser.add_argument(
        "--train_method",
        help="Method of training",
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
        help="Guidance scale for loss combination",
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
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        help="Output directory for the trained model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mask_path", help="Path to the mask file", type=str, required=True
    )
    parser.add_argument(
        "--device", help="Device to use for training", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--use_sample", action="store_true", help="Use the sample dataset for training"
    )

    # Device configuration
    parser.add_argument(
        "--devices",
        help="CUDA devices to train on (comma-separated)",
        type=str,
        required=False,
        default="0",
    )

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join(args.output_dir, "saliency_unlearning_training.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Saliency Unlearning Training")

    # Prepare output directory
    output_name = os.path.join(args.output_dir, f"{args.theme}.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving the model to {output_name}")

    # Set device
    devices = [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]

    config = {
        "train_method": args.train_method,
        "alpha": args.alpha,
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "original_data_dir": args.original_data_dir,
        "new_data_dir": args.new_data_dir,
        "output_dir": args.output_dir,
        "theme": args.theme,
        "class": args.classes,
        "devices": devices,
        "output_name": output_name,
        "mask_path": args.mask_path,
        "epochs": args.epochs,
        "use_sample": args.use_sample,
    }

    # Initialize and run the algorithm
    algorithm = SaliencyUnlearnAlgorithm(config)
    trained_model = algorithm.run()
