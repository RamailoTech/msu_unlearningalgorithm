# scripts/run_esd_training.py

import argparse
from utils.config_loader import load_config
from algorithms.esd.esd_algorithm import ESDAlgorithm

def main():
    parser = argparse.ArgumentParser(
        prog='TrainESD',
        description='Finetuning stable diffusion model to erase concepts using ESD method'
    )
    parser.add_argument('--train_method', help='method of training', type=str, default='noxattn', choices=['xattn', 'noxattn', 'selfattn', 'full'])
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, default='configs/train_esd.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=True)
    parser.add_argument('--devices', help='cuda devices to train on', type=str, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, default=50)
    parser.add_argument('--output_dir', help='output directory to save results', type=str, default='results/style50')
    parser.add_argument('--object_class', type=str, required=True)
    parser.add_argument('--dry_run', action='store_true', help='dry run')
    args = parser.parse_args()

    config = vars(args)  # Convert argparse.Namespace to dict

    # Construct the prompt
    config['prompt'] = f'An image of {args.object_class}.'

    algorithm = ESDAlgorithm(config)
    algorithm.run()

if __name__ == '__main__':
    main()
