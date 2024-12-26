import argparse
import os
import sys

import torch
from algorithms.saliency_unlearning.algorithm import MaskingAlgorithm


def main():
    parser = argparse.ArgumentParser(prog='GenerateMask', description='Generate saliency mask using MaskingAlgorithm.')

    parser.add_argument('--c_guidance', help='Guidance scale used in loss computation', type=float, default=7.5)
    parser.add_argument('--batch_size', help='Batch size used for mask generation', type=int, default=4)
    parser.add_argument('--ckpt_path', help='Checkpoint path for the model', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model configuration file')
    parser.add_argument('--num_timesteps', help='Number of timesteps for diffusion', type=int, default=1000)
    parser.add_argument('--theme', type=str, required=True, help='Concept or theme to unlearn')
    parser.add_argument('--classes', type=str, required=True, help='Class or objects to unlearn')
    parser.add_argument('--output_dir', help='Output directory for the generated mask', type=str, required=False, default='data/mask')
    parser.add_argument('--threshold', help='Threshold for mask generation', type=float, default=0.5)
    # Dataset directories
    parser.add_argument('--original_data_dir', type=str, required=False,default='/home/ubuntu/Projects/msu_unlearningalgorithm/data/quick-canvas-dataset/sample/quick-canvas-benchmark',
                        help='Directory containing the original dataset organized by themes and classes.')
    parser.add_argument('--new_data_dir', type=str, required=False,default='/home/ubuntu/Projects/msu_unlearningalgorithm/mu/algorithms/saliency_unlearning/data',
                        help='Directory where the new datasets will be saved.')

    parser.add_argument('--image_size', help='Image size for training', type=int, default=512)
    parser.add_argument('--lr', help='Learning rate for optimizer', type=float, default=1e-5)
    parser.add_argument('--device', help='Device to use for training', type=str, default='cuda:0')
    parser.add_argument('--use_sample', action='store_true', help='Use the sample dataset for training')

    args = parser.parse_args()

    prompt = f"An image in {args.theme} Style."
    output_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f'{args.threshold}.pt')
    if os.path.exists(mask_path):
        print(f"Mask for threshold {args.threshold} already exists at {mask_path}. Skipping.")
        return
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = {
        'original_data_dir': args.original_data_dir,
        'new_data_dir': args.new_data_dir,
        'mask_path': None,
        'theme': args.theme,
        'class': args.classes,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'interpolation': 'bicubic',
        'use_sample': False,
        'num_workers': 4,
        'pin_memory': True,
        'train_method': 'xattn',  # or whatever method needed
        'lr': args.lr,
        'config_path': args.config_path,
        'ckpt_path': args.ckpt_path,
        'output_dir': output_dir,
        'devices': [args.device],
        'c_guidance': args.c_guidance,
        'num_timesteps': args.num_timesteps,
        'threshold': args.threshold,
        'prompt': prompt,
        'use_sample': args.use_sample

    }

    algorithm = MaskingAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
