# mu/algorithms/forget_me_not/scripts/train_ti.py


import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import logs_dir


def main():
    parser = argparse.ArgumentParser(description="Forget Me Not - Train TI")

    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
        type=str,
        required=True,
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

    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", help="Learning rate used to train", type=float)

    parser.add_argument("--use_sample", help="Use the sample dataset for training")

    parser.add_argument(
        "--devices", type=str, help="CUDA devices to train on (comma-separated)"
    )

    parser.add_argument("--tokenizer_name", type=str, required=False)
    parser.add_argument("--instance_prompt", type=str, required=False)
    parser.add_argument("--concept_keyword", type=str, required=False)
    parser.add_argument("--lr_scheduler", type=str, required=False)
    parser.add_argument("--prior_generation_precision", type=str, required=False)
    parser.add_argument("--local_rank", type=int, required=False)
    parser.add_argument("--class_prompt", type=str, required=False)
    parser.add_argument("--num_class_images", type=int, required=False)
    parser.add_argument("--dataloader_num_workers", type=int, required=False)
    parser.add_argument("--center_crop", type=bool, required=False)
    parser.add_argument("--prior_loss_weight", type=float, required=False)
    parser.add_argument("--lr_warmup_steps", type=int, required=False)

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
        f"forget_me_not_training_ti_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Forget Me Not Training ti")

    # Initialize and run the EraseDiff algorithm
    algorithm = ForgetMeNotAlgorithm(config, train_type="train_ti")
    algorithm.run(train_type="train_ti")


if __name__ == "__main__":
    main()
