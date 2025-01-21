# mu/algorithms/concept_ablation/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.concept_ablation.algorithm import ConceptAblationAlgorithm
from mu.helpers import setup_logger, load_config, str2bool
from mu.helpers.path_setup import logs_dir
from mu.algorithms.concept_ablation.configs import ConceptAblationConfig


def main():
    parser = argparse.ArgumentParser(
        prog="TrainConceptAblation",
        description="Finetuning Stable Diffusion model to erase concepts using the Concept Ablation method",
    )

    parser.add_argument(
        "--config_path",
        help="Config path for Stable Diffusion",
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

    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="token added before cateogry word for personalization use case",
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default=None,
        help="crossattn to enable fine-tuning of all key, value, query matrices",
    )
    parser.add_argument(
        "--loss_type_reverse",
        type=str,
        default="model-based",
        help="loss type for reverse fine-tuning",
    )
    parser.add_argument(
        "--caption_target",
        type=str,
        help="target style to remove, used when kldiv loss",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_caption",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--datapath2",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_datapath2",
        type=str,
        default=None,
        help="path to regularization images",
    )
    parser.add_argument(
        "--caption2",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_caption2",
        type=str,
        default="",
        help="path to regularization images' caption",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="repeat the target dataset by how many times. Used when training without regularization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="overwrite batch size",
    )
    parser.add_argument("--base_lr", type=float, help="overwrite base learning rate")

    parser.add_argument(
        "--save_freq",
        type=int,
        default=None,
        help="overwrite every_n_train_steps in model saving callback",
    )
    parser.add_argument(
        "--image_logging_freq",
        type=int,
        default=None,
        help="overwrite batch_frequency in image logging callback",
    )
    parser.add_argument(
        "--train_max_steps",
        type=int,
        default=None,
        help="overwrite max_steps in finetuning",
    )
    parser.add_argument(
        "--parameter_group",
        type=str,
        default=None,
        choices=["full-weight", "cross-attn", "embedding"],
        help="parameter groups to finetune. Default: full-weight for memorization and cross-attn for others",
    )

    parser.add_argument(
        "--prompts", type=str, default=None, help="the initial path to ablation prompts"
    )
    parser.add_argument("--train_size", type=int, help="the number of generated images")

    parser.add_argument(
        "--n_samples", type=int, help="number of batch size in image generation"
    )
    parser.add_argument("--regularization", help="If True, add regularization loss")

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)

    # Prepare output directory
    os.makedirs(args.output_dir or config.get("output_dir", "results"), exist_ok=True)

    # Update configuration only if arguments are explicitly provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    config["lr"] = float(config["lr"])

    # Setup logger
    log_file = os.path.join(
        logs_dir,
        f"concept_ablation_training_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Concept Ablation Training")

    # Initialize and run the EraseDiff algorithm
    config = ConceptAblationConfig(**config)
    algorithm = ConceptAblationAlgorithm(config, config_path=args.config_path)
    algorithm.run()


if __name__ == "__main__":
    main()
