# main_debiasing.py

import argparse
from unified_concept_algorithm import UnifiedConceptEditingAlgorithm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Concept Editing - Debiasing Approach')
    parser.add_argument('--concepts', type=str, required=True, help='Comma-separated concepts to debias')
    parser.add_argument('--attributes', type=str, default='male,female', help='Comma-separated attributes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--retain_texts', type=str, default='', help='Comma-separated retain texts')

    args = parser.parse_args()

    concepts = args.concepts.split(',')
    attributes = args.attributes.split(',')
    retain_texts = args.retain_texts.split(',') if args.retain_texts else []

    old_texts = [f'image of {concept.strip()}' for concept in concepts]
    new_texts_list = [[text.replace(concept.strip(), attr.strip()) for attr in attributes] for text in old_texts for concept in concepts]

    config = {
        'ckpt_path': args.ckpt,
        'device': 'cuda',
        'old_texts': old_texts,
        'new_texts': new_texts_list,
        'retain_texts': retain_texts,
        'lamb': 0.5,
        'erase_scale': 1.0,
        'preserve_scale': 0.1,
        'output_path': args.output_dir,
        'with_to_k': True,
        'technique': 'replace',
        'approach': 'debiasing'
    }

    algorithm = UnifiedConceptEditingAlgorithm(config)
    algorithm.run()
