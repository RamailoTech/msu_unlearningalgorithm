# run_erasediff_training.py

import argparse
from utils.config_loader import load_config
from algorithms.erasediff.erasediff_algorithm import EraseDiffAlgorithm
import os
import torch

def main():
    parser = argparse.ArgumentParser(
        prog='EDiff',
        description='Finetuning stable diffusion model to erase concepts using EraseDiff'
    )

    parser.add_argument('--forget_data_dir', type=str, default='data')
    parser.add_argument('--remain_data_dir', type=str, default='data/Seed_Images/')
    parser.add_argument('--theme', type=str, required=True)
    parser.add_argument('--train_method', type=str, default="xattn",
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--config_path', type=str, default='configs/train_ediff.yaml')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--K_steps', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--dry_run', action='store_true', help='Dry run without logging to wandb')
    args = parser.parse_args()

    config = vars(args)  # Convert argparse.Namespace to dict
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['forget_data_dir'] = os.path.join(args.forget_data_dir, args.theme)
    config['output_dir'] = os.path.join(args.output_dir, args.theme)
    os.makedirs(config['output_dir'], exist_ok=True)

    algorithm = EraseDiffAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
