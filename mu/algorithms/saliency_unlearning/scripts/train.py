# scripts/train_saliency_unlearn.py

import os
import sys
import torch
from tqdm import tqdm
import argparse
import logging

from algorithms.saliency_unlearning.algorithm import SaliencyUnlearnAlgorithm
from algorithms.saliency_unlearning.utils import load_model_from_config
from datasets.constants import theme_available, class_available

def saliency_unlearning_train(forget_data_dir, remain_data_dir, output_dir, config_path, ckpt_path, mask_path,
                              train_method, alpha=0.1, batch_size=4, epochs=1, lr=1e-5, device="cuda:0",
                              image_size=512):
    """
    Function to execute the saliency unlearning training process.
    """
    # Initialize configuration
    config = {
        'original_data_dir': forget_data_dir,
        'new_data_dir': remain_data_dir,
        'mask_path': mask_path,
        'theme': train_method,
        'class': train_method,  # Adjust based on your use case
        'batch_size': batch_size,
        'image_size': image_size,
        'interpolation': 'bicubic',
        'use_sample': False,
        'num_workers': 4,
        'pin_memory': True,
        'train_method': train_method,
        'lr': lr,
        'epochs': epochs,
        'config_path': config_path,
        'ckpt_path': ckpt_path,
        'output_name': os.path.join(output_dir, f"saliency_unlearn_model_{train_method}.pth"),
        'devices': [device],
        'alpha': alpha,
        'K_steps': 2
    }

    # Initialize and run the algorithm
    algorithm = SaliencyUnlearnAlgorithm(config)
    trained_model = algorithm.run()

    return trained_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='SaliencyUnlearnTrain',
        description='Finetuning stable diffusion model to perform saliency-based unlearning.')

    parser.add_argument('--forget_data_dir', help='Directory containing forget data', type=str, required=True)
    parser.add_argument('--remain_data_dir', help='Directory containing remain data', type=str, required=True, default='data/Seed_Images/')
    parser.add_argument('--theme', help='Theme used for training', type=str, required=True, choices=theme_available + class_available)
    parser.add_argument('--train_method', help='Method of training', type=str, default="xattn",
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--alpha', help='Guidance scale for loss combination', type=float, required=False, default=0.1)
    parser.add_argument('--epochs', help='Number of epochs to train', type=int, required=False, default=1)
    parser.add_argument('--config_path', type=str, required=False, default='configs/train_saliency_unlearn.yaml')
    parser.add_argument('--ckpt_path', type=str, required=False, default='path/to/checkpoint.ckpt')
    parser.add_argument('--output_dir', help='Output directory for the trained model', type=str, required=True)
    parser.add_argument('--mask_path', help='Path to the mask file', type=str, required=True)
    parser.add_argument('--device', help='Device to use for training', type=str, default='cuda:0')
    parser.add_argument('--image_size', help='Image size for training', type=int, default=512)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Execute training
    trained_model = saliency_unlearning_train(
        forget_data_dir=args.forget_data_dir,
        remain_data_dir=args.remain_data_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        mask_path=args.mask_path,
        train_method=args.train_method,
        alpha=args.alpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        image_size=args.image_size
    )

    # Save the trained model
    torch.save({"state_dict": trained_model.state_dict()}, os.path.join(args.output_dir, f"saliency_unlearn_model_{args.train_method}.pth"))
