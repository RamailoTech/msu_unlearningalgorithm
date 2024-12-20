import argparse
import os
import logging
import sys

# Adjust these imports according to your project structure
from algorithms.concept_ablation.algorithm import ConceptAblationAlgorithm
from algorithms.erase_diff.logger import setup_logger  # or create a similar logger for concept_ablation if needed

def main():
    parser = argparse.ArgumentParser(
        prog='TrainConceptAblation',
        description='Fine-tuning a Stable Diffusion model to remove a certain concept using Concept Ablation.'
    )

    # Training parameters
    parser.add_argument('--train_method', type=str, default="full",
                        choices=["full", "xattn", "selfattn", "noxattn", "xlayer", "selflayer"],
                        help='Method of training the model parameters.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter if needed for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training.')

    # Model configuration
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model config file (YAML).')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--delta_ckpt', type=str, default=None, help='Path to a delta checkpoint, if any.')

    # Concept-related arguments
    parser.add_argument('--concept_type', type=str, required=True, choices=['style', 'object', 'memorization'],
                        help='Type of concept to remove.')
    parser.add_argument('--prompts_path', type=str, required=True,
                        help='Path to the file containing initial prompts for concept ablation.')
    parser.add_argument('--caption_target', type=str, default=None,
                        help='Target concept/style to remove, used especially for filtering in memorization tasks.')

    # Data output configurations
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store results.')
    parser.add_argument('--train_size', type=int, default=1000, help='Number of generated training images.')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of images per prompt batch generation step.')

    # Image and sampling configurations
    parser.add_argument('--image_size', type=int, default=512, help='Image size for training.')
    parser.add_argument('--interpolation', type=str, default='bicubic', choices=['bilinear', 'bicubic', 'lanczos'],
                        help='Interpolation mode for image resizing.')

    # Device configuration
    parser.add_argument('--devices', type=str, default='0', help='CUDA devices to train on (comma-separated).')

    # DataLoader configs
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading.')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Use pinned memory in DataLoader.')

    # Regularization option
    parser.add_argument('--use_regularization', action='store_true', help='Use a regularization dataset if applicable.')

    # Logging and run info
    parser.add_argument('--project_name', type=str, default='concept_ablation_project', help='WandB project name.')
    parser.add_argument('--run_name', type=str, default='concept_ablation_run', help='WandB run name.')
    parser.add_argument('--output_name', type=str, default='concept_ablation_model.pth', help='Output model filename.')

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join(args.output_dir, "concept_ablation_training.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Concept Ablation Training")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving results to {args.output_dir}")

    # Parse devices
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]

    # Create configuration dictionary
    config = {
        'train_method': args.train_method,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'lr': args.lr,
        'config_path': args.config_path,
        'ckpt_path': args.ckpt_path,
        'delta_ckpt': args.delta_ckpt,
        'concept_type': args.concept_type,
        'prompts_path': args.prompts_path,
        'caption_target': args.caption_target,
        'output_dir': args.output_dir,
        'train_size': args.train_size,
        'n_samples': args.n_samples,
        'image_size': args.image_size,
        'interpolation': args.interpolation,
        'devices': devices,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'use_regularization': args.use_regularization,
        'project_name': args.project_name,
        'run_name': args.run_name,
        'output_name': args.output_name
    }

    # Initialize and run the Concept Ablation algorithm
    algorithm = ConceptAblationAlgorithm(config)
    algorithm.run()


if __name__ == '__main__':
    main()
