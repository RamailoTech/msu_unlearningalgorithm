#scripts/evaluate.py

import argparse

from mu_attack.evaluators.asr import ASREvaluator
from mu_attack.evaluators.clip_score import ClipScoreEvaluator
from mu_attack.evaluators.fid import FIDEvaluator
from mu_attack.helpers.utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Evaluate metrics')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    
    # Override arguments for ASR
    parser.add_argument('--asr_root', type=str, help='Override ASR root directory')
    parser.add_argument('--asr_root_no_attack', type=str, help='Override ASR root-no-attack directory')
    
    # Override arguments for CLIP
    parser.add_argument('--devices', type=str, help="device to use (0,1)")
    parser.add_argument('--clip_image_path', type=str, help='Override CLIP image path')
    parser.add_argument('--clip_devices', type=str, help='Override CLIP devices (comma-separated)')
    parser.add_argument('--clip_prompt', type=str, help='Override CLIP prompt')
    parser.add_argument('--clip_model_name_or_path', type=str, help='Override CLIP model name or path')
    
    # Override arguments for FID
    parser.add_argument('--fid_ref_batch_path', type=str, help='Override FID reference batch path')
    parser.add_argument('--fid_sample_batch_path', type=str, help='Override FID sample batch path')
    
    args = parser.parse_args()

    config = load_config(args.config_path)

    devices = config.get('clip', {}).get('devices')
    devices = (
        [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
        if args.devices
        else [f'cuda:{int(d.strip())}' for d in devices.split(',')]
    )
    # Override ASR Config
    if args.asr_root:
        config['asr']['root'] = args.asr_root
    if args.asr_root_no_attack:
        config['asr']['root-no-attack'] = args.asr_root_no_attack

    # Override CLIP Config
    if args.clip_image_path:
        config['clip']['image_path'] = args.clip_image_path
    if args.clip_devices:
        config['clip']['devices'] = devices
    if args.clip_prompt:
        config['clip']['prompt'] = args.clip_prompt
    if args.clip_model_name_or_path:
        config['clip']['model_name_or_path'] = args.clip_model_name_or_path

    # Override FID Config
    if args.fid_ref_batch_path:
        config['fid']['ref_batch_path'] = args.fid_ref_batch_path
    if args.fid_sample_batch_path:
        config['fid']['sample_batch_path'] = args.fid_sample_batch_path

    config['clip']['devices'] = devices

    #Run the asr evaluator
    asr_evaluator = ASREvaluator(config)
    asr_evaluator.run()


    # Run the clip score evaluator
    clip_score_evaluator = ClipScoreEvaluator(config)
    clip_score_evaluator.run()

    # Run the fid evaluator
    fid_evaluator = FIDEvaluator(config)
    fid_evaluator.run()

if __name__ == '__main__':
    main()