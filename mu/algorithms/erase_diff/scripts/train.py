# mu/algorithms/erase_diff/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.erase_diff import EraseDiffAlgorithm
from mu.helpers import setup_logger, load_config, setup_logger

def main():
    parser = argparse.ArgumentParser(
        prog='TrainEraseDiff',
        description='Finetuning Stable Diffusion model to erase concepts using the EraseDiff method'
    )

    parser.add_argument('--config_path', help='Config path for Stable Diffusion', type=str,
                        required=False, default='configs/train_erase_diff.yaml')
    
    # Training parameters
    parser.add_argument('--train_method', help='method of training', type=str, default="xattn",
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--alpha', help='Guidance of start image used to train', type=float, required=False, default=0.1)
    parser.add_argument('--epochs', help='Number of epochs to train', type=int, required=False, default=1)
    parser.add_argument('--K_steps', type=int, required=False, default=2)
    parser.add_argument('--lr', help='Learning rate used to train', type=float, required=False, default=5e-5)

    # Model configuration
    parser.add_argument('--model_config_path', help='Model Config path for Stable Diffusion', type=str,
                        required=False, default='configs/train_erase_diff.yaml')
    parser.add_argument('--ckpt_path', help='Checkpoint path for Stable Diffusion', type=str, required=False,
                        default='path/to/checkpoint.ckpt')

    # Dataset directories
    parser.add_argument('--raw_dataset_dir', type=str, required=False,default='/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/quick-canvas-benchmark',
                        help='Directory containing the original dataset organized by themes and classes.')
    parser.add_argument('--processed_dataset_dir', type=str, required=False,default='/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/erase_diff/data',
                        help='Directory where the new datasets will be saved.')
    parser.add_argument('--dataset_type', type=str, required=False, default='unlearncanvas',
                        choices=['unlearncanvas', 'i2p'])
    parser.add_argument('--template', type=str, required=False, default='style',
                        choices=['object', 'style', 'i2p'])
    parser.add_argument('--template_name', type=str, required=False, default='self-harm',
                        choices=['self-harm', 'Abstractionism'])

    # Output configurations
    parser.add_argument('--output_dir', help='Output directory to save results', type=str, required=False,
                        default='results')
    parser.add_argument('--separator', help='Separator if you want to train multiple words separately', type=str,
                        required=False, default=None)

    # Sampling and image configurations
    parser.add_argument('--image_size', help='Image size used to train', type=int, required=False, default=512)
    parser.add_argument('--interpolation', help='Interpolation mode', type=str, required=False, default='bicubic',
                        choices=['bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--ddim_steps', help='DDIM steps of inference used to train', type=int, required=False,
                        default=50)
    parser.add_argument('--ddim_eta', help='DDIM eta parameter', type=float, required=False, default=0.0)

    # Device configuration
    parser.add_argument('--devices', help='CUDA devices to train on (comma-separated)', type=str, required=False, default='0')

    # Additional flags
    parser.add_argument('--use_sample', action='store_true', help='Use the sample dataset for training')
    parser.add_argument('--num_workers', type=int, required=False, default=4)
    parser.add_argument('--pin_memory', type=bool, required=False, default=True)

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)

    # Setup logger
    log_file = os.path.join(args.output_dir, "erase_diff_training.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting EraseDiff Training")

    # Prepare output directory
    output_name = os.path.join(args.output_dir, f"{args.theme}.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving the model to {output_name}")

    # Parse devices
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    # Create configuration dictionary
    config.update({
        'train_method': args.train_method or config.get('train_method', 'xattn'),
        'alpha': args.alpha or config.get('alpha', 0.1),
        'epochs': args.epochs or config.get('epochs', 1),
        'K_steps': args.K_steps or config.get('K_steps', 2),
        'lr': args.lr or config.get('lr', 5e-5),
        'model_config_path': args.model_config_path or config.get('model_config_path', 'configs/train_erase_diff.yaml'),
        'ckpt_path': args.ckpt_path or config.get('ckpt_path', 'path/to/checkpoint.ckpt'),
        'raw_dataset_dir': args.raw_dataset_dir or config.get('raw_dataset_dir'),
        'processed_dataset_dir': args.processed_dataset_dir or config.get('processed_dataset_dir'),
        'dataset_type': args.dataset_type or config.get('dataset_type', 'unlearncanvas'),
        'template': args.template or config.get('template', 'style'),
        'template_name': args.template_name or config.get('template_name', 'self-harm'),
        'output_dir': args.output_dir or config.get('output_dir', 'results'),
        'separator': args.separator or config.get('separator'),
        'image_size': args.image_size or config.get('image_size', 512),
        'interpolation': args.interpolation or config.get('interpolation', 'bicubic'),
        'ddim_steps': args.ddim_steps or config.get('ddim_steps', 50),
        'ddim_eta': args.ddim_eta or config.get('ddim_eta', 0.0),
        'devices': devices or config.get('devices', ['cuda:0']),
        'num_workers': args.num_workers or config.get('num_workers', 4),
        'pin_memory': args.pin_memory or config.get('pin_memory', True),
    })

    # Initialize and run the EraseDiff algorithm
    algorithm = EraseDiffAlgorithm(config)
    algorithm.run()


if __name__ == '__main__':
    main()
