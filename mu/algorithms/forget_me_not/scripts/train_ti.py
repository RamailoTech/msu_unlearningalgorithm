# forget_me_not/scripts/train_ti.py

import argparse
import os
import yaml
from algorithms.forget_me_not.algorithm import ForgetMeNotAlgorithm

def main():
    parser = argparse.ArgumentParser(description='Forget Me Not - Train TI')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the pretrained diffuser directory.')
    parser.add_argument('--theme', type=str, required=True, help='Theme or concept to unlearn.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs and checkpoints.')
    parser.add_argument('--steps', type=int, default=500, help='Number of training steps for TI.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for TI training.')
    parser.add_argument('--instance_data_dir', type=str, default='data', help='Directory containing instance images.')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging.')
    parser.add_argument('--wandb_project', type=str, default='forget_me_not', help='WandB project name.')
    parser.add_argument('--wandb_name', type=str, default='ti_run', help='WandB run name.')

    # Dataset directories
    parser.add_argument('--original_data_dir', type=str, required=False,default='/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/quick-canvas-benchmark',
                        help='Directory containing the original dataset organized by themes and classes.')
    parser.add_argument('--new_data_dir', type=str, required=False,default='/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/erase_diff/data',
                        help='Directory where the new datasets will be saved.')
    

    args = parser.parse_args()

    # Construct configuration dictionary
    config = {
        'pretrained_model_name_or_path': args.pretrained_path,
        'theme': args.theme,
        'output_dir': os.path.join(args.output_dir, args.theme),
        'steps': args.steps,
        'lr': args.lr,
        'instance_data_dir': os.path.join(args.instance_data_dir, args.theme),
        'train_batch_size': 1,
        'save_steps': 500,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_name': args.wandb_name,
        # Add any additional configurations needed by your data handler, model, or trainer.
    }

    # Initialize and run the ForgetMeNotAlgorithm for TI training
    algorithm = ForgetMeNotAlgorithm(config)
    algorithm.run_ti_training()

if __name__ == '__main__':
    main()
