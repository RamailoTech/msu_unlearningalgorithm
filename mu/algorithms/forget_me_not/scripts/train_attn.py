# mu/algorithms/forget_me_not/scripts/train_attn.py

import argparse
import os
from pathlib import Path
import logging


from mu.algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm
from mu.helpers import setup_logger, load_config
from mu.helpers.path_setup import logs_dir



def main():
    parser = argparse.ArgumentParser(description='Forget Me Not - Train TI')
    
    parser.add_argument('--config_path', help='Config path for Stable Diffusion', type=str,
                        required=True)
    
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
    parser.add_argument('--output_dir', help='Output directory to save results', type=str)

    parser.add_argument('--ti_weights_path', help='Train inversion model weights', type=str)
    parser.add_argument('--lr', help='Learning rate used to train', type=float)
    parser.add_argument('--use_sample', help='Use the sample dataset for training')
    parser.add_argument('--devices', type=str, help='CUDA devices to train on (comma-separated)')
    

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)


    # Prepare output directory
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
    config['type'] = 'train_attn'

    # Setup logger
    log_file = os.path.join(logs_dir, f"forget_me_not_training_attn_{config.get('dataset_type')}_{config.get('template')}_{config.get('template_name')}.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Forget Me Not Training attn")

    # Initialize and run the EraseDiff algorithm
    algorithm = ForgetMeNotAlgorithm(config)
    algorithm.run(train_type='train_attn')

if __name__ == '__main__':
    main()