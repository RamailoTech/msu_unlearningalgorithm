# main_erasing.py

import argparse
from unified_concept_algorithm import UnifiedConceptEditingAlgorithm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Concept Editing - Erasing Approach')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--theme', type=str, required=True, help='Theme to erase')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--retain_texts', type=str, default='', help='Comma-separated retain texts')

    args = parser.parse_args()

    retain_texts = args.retain_texts.split(',') if args.retain_texts else []

    config = {
        'ckpt_path': args.ckpt,
        'device': 'cuda',
        'old_texts': [args.theme],
        'new_texts': [' '],  # Erasing by replacing with space
        'retain_texts': retain_texts,
        'lamb': 0.5,
        'erase_scale': 1.0,
        'preserve_scale': 0.1,
        'output_path': args.output_dir,
        'with_to_k': True,
        'technique': 'replace',
        'approach': 'erasing'
    }

    algorithm = UnifiedConceptEditingAlgorithm(config)
    algorithm.run()
