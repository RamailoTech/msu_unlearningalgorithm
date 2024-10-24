# main.py

import argparse
from saliency_unlearning_algorithm import SaliencyUnlearningAlgorithm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saliency Unlearning Algorithm')

    parser.add_argument('--forget_data_dir', type=str, required=True)
    parser.add_argument('--remain_data_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--c_guidance', type=float, default=1.0)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()

    config = {
        'forget_data_dir': args.forget_data_dir,
        'remain_data_dir': args.remain_data_dir,
        'config_path': args.config_path,
        'ckpt_path': args.ckpt_path,
        'output_path': args.output_path,
        'prompt': args.prompt,
        'device': args.device,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'c_guidance': args.c_guidance,
        'num_timesteps': args.num_timesteps,
        'threshold': args.threshold,
        'epochs': args.epochs,
        'alpha': args.alpha,
        'lr': args.lr
    }

    algorithm = SaliencyUnlearningAlgorithm(config)
    algorithm.run()
