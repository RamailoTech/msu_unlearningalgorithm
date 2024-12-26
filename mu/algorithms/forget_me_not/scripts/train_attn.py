# forget_me_not/scripts/train_attn.py

import argparse
import os

from algorithms.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm


def main():
    parser = argparse.ArgumentParser(description="Forget Me Not - Train Attention")
    parser.add_argument(
        "--theme", type=str, required=True, help="Theme or concept to unlearn."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for attention-based training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of training steps for attention training.",
    )
    parser.add_argument(
        "--ti_weight_path",
        type=str,
        required=True,
        help="Path to TI weights (e.g., from train_ti step).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs and checkpoints.",
    )
    parser.add_argument(
        "--only_xa",
        action="store_true",
        help="If set, only optimize cross-attention parameters.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="data",
        help="Directory containing instance images.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="",
        help="Path to the pretrained diffuser directory.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="forget_me_not", help="WandB project name."
    )
    parser.add_argument(
        "--wandb_name", type=str, default="attn_run", help="WandB run name."
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

    args = parser.parse_args()

    config = {
        "original_data_dir": args.original_data_dir,
        "new_data_dir": args.new_data_dir,
        "theme": args.theme,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "ti_weight_path": args.ti_weight_path,
        "output_dir": os.path.join(args.output_dir, args.theme),
        "instance_data_dir": os.path.join(args.instance_data_dir, args.theme),
        "train_batch_size": 1,
        "save_steps": 200,
        "pretrained_model_name_or_path": args.pretrained_path,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "only_optimize_ca": args.only_xa,
        # Add any additional configurations needed by your data handler, model, or trainer.
    }

    # Initialize and run the ForgetMeNotAlgorithm for attention training
    algorithm = ForgetMeNotAlgorithm(config)
    algorithm.run_attn_training()


if __name__ == "__main__":
    main()
