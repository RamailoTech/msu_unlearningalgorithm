# scripts/train.py

import argparse
import os

from algorithms.esd.algorithm import ESDAlgorithm

def main():
    parser = argparse.ArgumentParser(
        prog='TrainESD',
        description='Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--train_method', help='method of training', type=str, required=True,
                        choices=['xattn', 'noxattn', 'selfattn', 'full'])
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float,
                        required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float,
                        required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion', type=str,
                        required=False, default='configs/train_esd.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion', type=str, required=False,
                        default='path/to/checkpoint.ckpt')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train a bunch of words separately', type=str,
                        required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=50)
    parser.add_argument('--output_dir', help='output directory to save results', type=str, required=False,
                        default='results')
    parser.add_argument('--theme', type=str, required=True, help='Concept or theme to unlearn')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f'{args.output_dir}/{args.theme}.pth'
    print(f"Saving the model to {output_name}")

    prompt = f'{args.theme.replace("_", " ")} Style'
    print(f"Prompt for unlearning: {prompt}")

    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]

    config = {
        'train_method': args.train_method,
        'start_guidance': args.start_guidance,
        'negative_guidance': args.negative_guidance,
        'iterations': args.iterations,
        'lr': args.lr,
        'config_path': args.config_path,
        'ckpt_path': args.ckpt_path,
        'devices': devices,
        'seperator': args.seperator,
        'image_size': args.image_size,
        'ddim_steps': args.ddim_steps,
        'output_name': output_name,
        'prompt': prompt,
        'theme': args.theme,
    }

    algorithm = ESDAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
