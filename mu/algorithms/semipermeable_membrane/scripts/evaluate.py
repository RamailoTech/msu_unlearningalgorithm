from argparse import ArgumentParser
from mu.helpers import load_config
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
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Precision for the base model.",
    )
    parser.add_argument('--model_config', type=str, help="Path for model_config")
    parser.add_argument('--base_model', type=str, help="base model path")
    parser.add_argument('--theme', type=str, help="theme")
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

    #  Override config fields if CLI arguments are provided
    if args.model_config is not None:
        config["model_config"] = args.model_config
    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path
    if args.theme is not None:
        config["theme"] = args.theme
    if args.cfg_text_list is not None:
        config["cfg_text_list: [9.0]"] = args.cfg_text_list
    if args.seed is not None:
        config["seed"] = args.seed
    if args.ddim_steps is not None:
        config["ddim_steps"] = args.ddim_steps
    if args.image_height is not None:
        config["image_height"] = args.image_height
    if args.image_width is not None:
        config["image_width"] = args.image_width
    if args.ddim_eta is not None:
        config["ddim_eta"] = args.ddim_eta
    if args.sampler_output_dir is not None:
        config["sampler_output_dir"] = args.sampler_output_dir
    if args.classification_model is not None:
        config["classification_model"] = args.classification_model
    if args.eval_output_dir is not None:
        config["eval_output_dir"] = args.eval_output_dir
    if args.reference_dir is not None:
        config["reference_dir"] = args.reference_dir
    if args.forget_theme is not None:
        config["forget_theme"] = args.forget_theme
    if args.multiprocessing is not None:
        config["multiprocessing"] = args.multiprocessing
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    evaluator = SemipermeableMembraneEvaluator(config)
    evaluator.run()

if __name__ == "__main__":
    main()
