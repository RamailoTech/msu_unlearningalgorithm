# mu/algorithms/forget_me_not/scripts/train_attn.py

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

    parser.add_argument(
        "--ti_weights_path", help="Train inversion model weights", type=str
    )
    parser.add_argument("--lr", help="Learning rate used to train", type=float)
    parser.add_argument("--use_sample", help="Use the sample dataset for training")
    parser.add_argument(
        "--devices", type=str, help="CUDA devices to train on (comma-separated)"
    )

    # Newly added parameters
    parser.add_argument("--train_text_encoder", type=bool, required=False)
    parser.add_argument("--perform_inversion", type=bool, required=False)
    parser.add_argument("--continue_inversion", type=bool, required=False)
    parser.add_argument("--continue_inversion_lr", type=float, required=False)
    parser.add_argument("--learning_rate_ti", type=float, required=False)
    parser.add_argument("--learning_rate_unet", type=float, required=False)
    parser.add_argument("--learning_rate_text", type=float, required=False)
    parser.add_argument("--lr_scheduler", type=str, required=False)
    parser.add_argument("--lr_scheduler_lora", type=str, required=False)
    parser.add_argument("--lr_warmup_steps_lora", type=int, required=False)
    parser.add_argument("--prior_loss_weight", type=float, required=False)
    parser.add_argument("--weight_decay_lora", type=float, required=False)
    parser.add_argument("--mixed_precision", type=str, required=False)
    parser.add_argument("--use_8bit_adam", type=bool, required=False)
    parser.add_argument("--use_face_segmentation_condition", type=bool, required=False)
    parser.add_argument("--max_train_steps_ti", type=int, required=False)
    parser.add_argument("--max_train_steps_tuning", type=int, required=False)
    parser.add_argument("--save_steps", type=int, required=False)
    parser.add_argument("--class_data_dir", type=str, required=False)
    parser.add_argument("--stochastic_attribute", type=str, required=False)
    parser.add_argument("--class_prompt", type=str, required=False)
    parser.add_argument("--with_prior_preservation", type=bool, required=False)
    parser.add_argument("--num_class_images", type=int, required=False)
    parser.add_argument("--resolution", type=int, required=False)
    parser.add_argument("--color_jitter", type=bool, required=False)
    parser.add_argument("--sample_batch_size", type=int, required=False)
    parser.add_argument("--lora_rank", type=int, required=False)
    parser.add_argument("--clip_ti_decay", type=bool, required=False)

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
        f"forget_me_not_training_attn_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log",
    )
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Forget Me Not Training attn")

    # Initialize and run the EraseDiff algorithm
    algorithm = ForgetMeNotAlgorithm(config, train_type="train_attn")
    algorithm.run(train_type="train_attn")


if __name__ == "__main__":
    main()
