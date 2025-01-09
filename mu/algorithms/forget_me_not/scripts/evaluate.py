# mu/algorithms/forget_me_not/scripts/evaluate.py

import logging
import os
from argparse import ArgumentParser

from mu.helpers import load_config, setup_logger
from mu.helpers.path_setup import logs_dir
from mu.algorithms.forget_me_not import ForgetMeNotEvaluator


def main():
    """Main entry point for running the entire pipeline."""
    parser = ArgumentParser(description="Unified ConceptAblation Evaluation and Sampling")
    parser.add_argument('--config_path', required=True, help="Path to the YAML config file.")

    # Below: optional overrides for your config dictionary
    parser.add_argument('--devices', type=str, help='CUDA devices to train on (comma-separated)')
    parser.add_argument('--theme', type=str, help="theme")
    
    #model path
    parser.add_argument('--ckpt_path', type=str, help="checkpoint path")
    parser.add_argument('--classification_model', type=str, help="classification model name")

    #hyperparameters
    parser.add_argument('--cfg_text_list', type=float, help="(guidance scale)",nargs="+")
    parser.add_argument('--seed', type=int, help="seed")
    parser.add_argument('--ddim_steps', type=int, help="number of ddim_steps")
    parser.add_argument('--image_height', type=int, help="image height, in pixel space")
    parser.add_argument('--image_width', type=int, help="image width, in pixel space")
    parser.add_argument('--ddim_eta', type=float, help="ddim eta (eta=0.0 corresponds to deterministic sampling")

    #output dir
    parser.add_argument('--sampler_output_dir', type=str, help="output directory for sampler")
    parser.add_argument('--eval_output_dir', type=str, help="evaluation output directory")
    parser.add_argument('--reference_dir', type=str, help="reference images directory")

    parser.add_argument('--forget_theme', type=str, help="forget_theme setting")
    parser.add_argument('--multiprocessing', type=bool, help="multiprocessing flag (True/False)")
    parser.add_argument('--batch_size', type=int, help="FID batch_size")

    args = parser.parse_args()

    config = load_config(args.config_path)

    devices = (
        [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
        if args.devices
        else [f'cuda:{int(d.strip())}' for d in config.get('devices').split(',')]
    )

    #  Override config fields if CLI arguments are provided
    for key, value in vars(args).items():
        if value is not None:  # Update only if the argument is provided
            config[key] = value

    config['devices'] = devices

    log_file = os.path.join(logs_dir, f"forget_me_not_evalaute_{config.get('theme')}.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting Forget Me Not Evalution Framework")

    evaluator = ForgetMeNotEvaluator(config)
    evaluator.run()

if __name__ == "__main__":
    main()
