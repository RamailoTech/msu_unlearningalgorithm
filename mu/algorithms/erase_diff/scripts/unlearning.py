import os
import argparse
import torch
import wandb

from mu.algorithms.erase_diff.model import EraseDiffModel
from mu.algorithms.erase_diff.trainer import EraseDiffTrainer
from mu.datasets.generate_dataset import generate_dataset
from mu.helpers.config_loader import load_config
from mu.helpers.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(
        prog='EraseDiffTrainer',
        description='Finetuning stable diffusion model to erase concepts using EraseDiff'
    )

    parser.add_argument('--forget_data_dir', type=str, default='data/erase_diff',
                        help='Directory containing forget data organized by theme.')
    parser.add_argument('--remain_data_dir', type=str, default='data/Seed_Images',
                        help='Directory containing remain data.')
    parser.add_argument('--theme', type=str, required=True,
                        help='Theme used to train. Must be one of the available themes.')
    parser.add_argument('--train_method', type=str, default='xattn',
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"],
                        help='Method of training.')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Guidance coefficient for loss.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train.')
    parser.add_argument('--config_path', type=str, default='algorithms/erase_diff/config/train_config.yaml',
                        help='Path to the training configuration file.')
    parser.add_argument('--ckpt_path', type=str, default='path/to/erase_diff_ckpt.ckpt',
                        help='Path to the checkpoint file.')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for saving the trained model.')
    parser.add_argument('--K_steps', type=int, default=2,
                        help='Number of K steps.')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Setup output directory
    output_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger('EraseDiffTrainer', os.path.join(output_dir, 'training.log'))
    logger.info("Starting EraseDiff training.")

    # Load configuration
    config = load_config(args.config_path)
    config.update(vars(args))  # Override config with command-line arguments

    # Initialize WandB
    wandb.init(project="quick-canvas-machine-unlearning", name=args.theme, config=config)

    # Generate datasets if necessary
    generate_dataset(
        original_data_dir='../../data/quick-canvas-benchmark',
        new_dir=args.forget_data_dir,
        theme_available=[args.theme],
        class_available=config.get('class_available', []),
        seed_images_theme="Seed_Images",
        num_images_per_class=config.get('num_images_per_class', 3)
    )

    # Initialize trainer
    trainer = EraseDiffTrainer(
        config=config,
        device=device,
        device_orig='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # Start training
    trainer.train()

    # Save the trained model
    torch.save({"state_dict": trainer.model.model.state_dict()}, os.path.join(output_dir, "sd.ckpt"))
    logger.info(f"Model saved to {os.path.join(output_dir, 'sd.ckpt')}")
    wandb.finish()

if __name__ == '__main__':
    main()
