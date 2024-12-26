import argparse
import os

import yaml
from algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm


def load_config(yaml_path):
    """Loads the configuration from a YAML file."""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    return {}

def main():
    parser = argparse.ArgumentParser(description='Forget Me Not - Train TI')
    
    # Command-line arguments
    parser.add_argument('--config_path', type=str, default='config/train_ti.yaml',
                        help='Path to the configuration YAML file.')
    parser.add_argument('--pretrained_path', type=str, help='Path to the pretrained diffuser directory.')
    parser.add_argument('--theme', type=str, help='Theme or concept to unlearn.')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs and checkpoints.')
    parser.add_argument('--steps', type=int, help='Number of training steps for TI.')
    parser.add_argument('--lr', type=float, help='Learning rate for TI training.')
    parser.add_argument('--instance_data_dir', type=str, help='Directory containing instance images.')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging.')
    parser.add_argument('--wandb_project', type=str, help='WandB project name.')
    parser.add_argument('--wandb_name', type=str, help='WandB run name.')

    # Dataset directories
    parser.add_argument('--original_data_dir', type=str, help='Directory containing the original dataset organized by themes and classes.')
    parser.add_argument('--new_data_dir', type=str, help='Directory where the new datasets will be saved.')

    args = parser.parse_args()

    # Load default configuration from YAML
    config = load_config(args.config_path)

    # Override YAML configuration with command-line arguments if provided
    config.update({
        'pretrained_model_name_or_path': args.pretrained_path or config.get('pretrained_model_name_or_path'),
        'theme': args.theme or config.get('theme'),
        'output_dir': os.path.join(args.output_dir or config.get('output_dir', ''), args.theme or config.get('theme')),
        'steps': args.steps or config.get('steps', 500),
        'lr': args.lr or config.get('lr', 1e-4),
        'instance_data_dir': os.path.join(args.instance_data_dir or config.get('instance_data_dir', 'data'), args.theme or config.get('theme')),
        'train_batch_size': config.get('train_batch_size', 1),
        'save_steps': config.get('save_steps', 500),
        'use_wandb': args.use_wandb or config.get('use_wandb', False),
        'wandb_project': args.wandb_project or config.get('wandb_project', 'forget_me_not'),
        'wandb_name': args.wandb_name or config.get('wandb_name', 'ti_run'),
        'original_data_dir': args.original_data_dir or config.get('original_data_dir'),
        'new_data_dir': args.new_data_dir or config.get('new_data_dir'),
        'initializer_tokens': args.theme or config.get('initializer_tokens', args.theme)
    })

    # Initialize and run the ForgetMeNotAlgorithm for TI training
    algorithm = ForgetMeNotAlgorithm(config)
    algorithm.run_ti_training()

if __name__ == '__main__':
    main()
