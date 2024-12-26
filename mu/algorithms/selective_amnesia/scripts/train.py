import argparse
import logging
import os

from algorithms.erase_diff.logger import (
    setup_logger,  # or adapt a similar logger if needed.
)
from algorithms.selective_amnesia.algorithm import SelectiveAmnesiaAlgorithm


def main():
    parser = argparse.ArgumentParser(description='Train Selective Amnesia')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model config')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to SD v1.4 ckpt')
    parser.add_argument('--fim_path', type=str, required=True, help='Path to precomputed FIM (full_fisher_dict.pkl)')
    parser.add_argument('--surrogate_data_dir', type=str, required=True, help='Path to q(x|c_f) surrogate dataset')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--devices', type=str, default='0', help='CUDA devices (comma-separated)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--interpolation', type=str, default='bicubic', choices=['bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--train_method', type=str, default='full', 
                        choices=["full", "xattn", "selfattn", "noxattn", "xlayer", "selflayer"])
    parser.add_argument('--project_name', type=str, default='selective_amnesia')
    parser.add_argument('--run_name', type=str, default='selective_amnesia_run')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--output_name', type=str, default='selective_amnesia_model.pth')

    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "selective_amnesia_training.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Selective Amnesia Training")

    devices = [f'cuda:{d.strip()}' for d in args.devices.split(',')]

    config = {
        'config_path': args.config_path,
        'ckpt_path': args.ckpt_path,
        'fim_path': args.fim_path,
        'surrogate_data_dir': args.surrogate_data_dir,
        'output_dir': args.output_dir,
        'devices': devices,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'interpolation': args.interpolation,
        'train_method': args.train_method,
        'project_name': args.project_name,
        'run_name': args.run_name,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'output_name': args.output_name
    }

    algorithm = SelectiveAmnesiaAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
