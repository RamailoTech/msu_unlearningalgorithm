# mu/algorithms/saliency_unlearning/scripts/generate_mask.py

import argparse
import os
from pathlib import Path
import logging

from mu.algorithms.saliency_unlearning.algorithm import MaskingAlgorithm
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import *

def main():
    parser = argparse.ArgumentParser(prog='GenerateMask', description='Generate saliency mask using MaskingAlgorithm.')

    parser.add_argument('--config_path', help='Config path for Stable Diffusion', type=str,
                        required=True)
    
    parser.add_argument('--c_guidance', help='Guidance scale used in loss computation', type=float, )
    parser.add_argument('--batch_size', help='Batch size used for mask generation', type=int, )
    parser.add_argument('--ckpt_path', help='Checkpoint path for the model', type=str)
    parser.add_argument('--model_config_path', type=str, help='Path to the model configuration file')
    parser.add_argument('--num_timesteps', help='Number of timesteps for diffusion', type=int)


    # Dataset directories
    parser.add_argument('--raw_dataset_dir', type=str,
                        help='Directory containing the original dataset organized by themes and classes.')
    parser.add_argument('--processed_dataset_dir', type=str,
                        help='Directory where the new datasets will be saved.')
    parser.add_argument('--dataset_type', type=str, choices=['unlearncanvas', 'i2p'])
    parser.add_argument('--template', type=str, choices=['object', 'style', 'i2p'])
    parser.add_argument('--template_name', type=str, choices=['self-harm', 'Abstractionism'])

    parser.add_argument('--output_dir', help='Output directory for the masks ', type=str)
    parser.add_argument('--threshold', help='Threshold for mask generation', type=float)

    parser.add_argument('--image_size', help='Image size used to train', type=int)
    parser.add_argument('--lr', help='Learning rate used to train', type=float)
    parser.add_argument('--devices', help='CUDA devices to train on (comma-separated)', type=str)
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
    log_file = os.path.join(logs_dir, f"saliency_unlearning_masking_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Saliency Unlearning Masking")

    algorithm = MaskingAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
