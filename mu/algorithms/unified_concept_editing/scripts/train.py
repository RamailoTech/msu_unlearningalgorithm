# unified_concept_editing/scripts/train.py

import argparse
import os
import logging
import yaml

from algorithms.unified_concept_editing.algorithm import UnifiedConceptEditingAlgorithm
from algorithms.unified_concept_editing.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        prog='TrainUnifiedConceptEditing',
        description='Finetuning Stable Diffusion model for Unified Concept Editing using the UnifiedConceptEditing method'
    )

    # Configuration file
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration YAML file.')

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    os.makedirs(config['output_dir'], exist_ok=True)
    log_file = os.path.join(config['output_dir'], "unified_concept_editing_training.log")
    logger = setup_logger(name='UnifiedConceptEditingTraining', log_file=log_file, level=logging.INFO)
    logger.info("Starting Unified Concept Editing Training")

    # Initialize and run the UnifiedConceptEditing algorithm
    algorithm = UnifiedConceptEditingAlgorithm(config)
    algorithm.run()


if __name__ == '__main__':
    main()
