#mu/algorithms/semipermeable_membrane/scripts/evaluate.py

import os
import logging
from argparse import ArgumentParser

from mu.helpers import setup_logger,load_config
from mu.helpers.path_setup import logs_dir
from mu.algorithms.semipermeable_membrane import SemipermeableMembraneEvaluator

def main():
    """Main entry point for running the entire pipeline."""
    parser = ArgumentParser(description="Unified SemipermeableMembrane Evaluation and Sampling")
    parser.add_argument('--config_path', required=True, help="Path to the YAML config file.")

    # Below: optional overrides for your config dictionary
    parser.add_argument(
        "--spm_multiplier",
        nargs="*",
        type=float,
        help="Assign multipliers for SPM model or set to `None` to use Facilitated Transport.",
    )
    parser.add_argument(
        "--matching_metric",
        type=str,
        help="matching metric for prompt vs erased concept",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model for generation.",
    )
    parser.add_argument(
        "--spm_path",
        type=list,
        help="paths to model checkpoints",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Precision for the base model.",
    )
    parser.add_argument('--theme', type=str, help="theme")
    parser.add_argument('--devices', type=str, help="device")
    parser.add_argument('--seed', type=int, help="seed")
    parser.add_argument('--sampler_output_dir', type=str, help="output directory for sampler")
    parser.add_argument('--classification_model', type=str, help="classification model name")
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

    #logger setuup
    log_file = os.path.join(logs_dir, f"semipermeable_membrane_evalaute_{config.get('theme')}.log")
    logger = setup_logger(log_file=log_file, level=logging.INFO)
    logger.info("Starting semipermeable membrane Evalution Framework")

    evaluator = SemipermeableMembraneEvaluator(config)
    evaluator.run()

if __name__ == "__main__":
    main()