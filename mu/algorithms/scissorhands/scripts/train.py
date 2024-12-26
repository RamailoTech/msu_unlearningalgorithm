# mu/algorithms/scissorhands/scripts/train.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.scissorhands.algorithm import ScissorHandsAlgorithm
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import *

def main():
    parser = argparse.ArgumentParser(
        prog='TrainScissorHands',
        description='Finetuning Stable Diffusion model to erase concepts using the EraseDiff method'
    )

    parser.add_argument('--config_path', help='Config path for Stable Diffusion', type=str,
                        required=True)
    
    # Training parameters
    parser.add_argument('--train_method', help='method of training', type=str, default="xattn",
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--alpha', help='Guidance of start image used to train', type=float)
    parser.add_argument('--epochs', help='Number of epochs to train', type=int)

    # Model configuration
    parser.add_argument('--model_config_path', help='Model Config path for Stable Diffusion', type=str)
    parser.add_argument('--ckpt_path', help='Checkpoint path for Stable Diffusion', type=str)

    # Dataset directories
    parser.add_argument('--raw_dataset_dir', type=str,
                        help='Directory containing the original dataset organized by themes and classes.')
    parser.add_argument('--processed_dataset_dir', type=str,
                        help='Directory where the new datasets will be saved.')
    parser.add_argument('--dataset_type', type=str, choices=['unlearncanvas', 'i2p'])
    parser.add_argument('--template', type=str, choices=['object', 'style', 'i2p'])
    parser.add_argument('--template_name', type=str, choices=['self-harm', 'Abstractionism'])


    # Output configurations
    parser.add_argument('--output_dir', help='Output directory to save results', type=str, required=False,
                        default='results')
    
    # Sampling and image configurations
    parser.add_argument('--sparsity', help='threshold for mask', type=float)
    parser.add_argument('--project', action='store_true')
    parser.add_argument('--memory_num', type=int)
    parser.add_argument('--prune_num', type=int)

    # Device configuration
    parser.add_argument('--devices', help='CUDA devices to train on (comma-separated)', type=str)

    # Additional flags
    parser.add_argument('--use_sample', help='Use the sample dataset for training')

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)


    # Prepare output directory
    output_name = os.path.join(args.output_dir or config.get('output_dir', 'results'), f"{args.template_name or config.get('template_name', 'self-harm')}.pth")
    os.makedirs(args.output_dir or config.get('output_dir', 'results'), exist_ok=True)

    # Parse devices
    devices = (
        [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
        if args.devices
        else [f'cuda:{int(d.strip())}' for d in config.get('devices').split(',')]
    )

    # Update configuration only if arguments are explicitly provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    # Ensure devices are properly set
    config['devices'] = devices

    # Setup logger
    log_file = os.path.join(logs_dir, f"erase_diff_training_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting scissorhands Training")

    # Initialize and run the EraseDiff algorithm
    algorithm = ScissorHandsAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
