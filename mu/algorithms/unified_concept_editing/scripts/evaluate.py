from argparse import ArgumentParser
from mu.helpers import load_config
from mu.algorithms.unified_concept_editing import UnifiedConceptEditingEvaluator

def main():
    """Main entry point for running the entire pipeline."""
    parser = ArgumentParser(description="Unified Concept Editing Evaluation and Sampling")
    parser.add_argument('--config_path', required=True, help="Path to the YAML config file.")

    # Below: optional overrides for your config dictionary
    parser.add_argument('--ckpt_path', type=str, help="checkpoint path")
    parser.add_argument('--num_samples', help='number of samples to be generated (batch_size)', type=str)
    parser.add_argument('--pipeline_path', help='path to pipeline', type=str)
    parser.add_argument('--theme', type=str, help="theme", required=True)
    parser.add_argument('--cfg_text', type=float, help="(guidance scale)")
    parser.add_argument('--seed', type=int, help="seed",required=False)
    parser.add_argument('--ddim_steps', type=int, help="number of ddim_steps",required=False)
    parser.add_argument('--image_height', type=int, help="image height")
    parser.add_argument('--image_width', type=int, help="image width")
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
    if args.pipeline_path is not None:
        config["pipeline_path"] = args.pipeline_path
    if args.num_samples is not None:
        config["num_samples"] = args.num_samples
    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path
    if args.theme is not None:
        config["theme"] = args.theme
    if args.cfg_text is not None:
        config["cfg_text"] = args.cfg_text
    if args.seed is not None:
        config["seed"] = args.seed
    if args.ddim_steps is not None:
        config["ddim_steps"] = args.ddim_steps
    if args.image_height is not None:
        config["image_height"] = args.image_height
    if args.image_width is not None:
        config["image_width"] = args.image_width
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

    evaluator = UnifiedConceptEditingEvaluator(config)
    evaluator.run()

if __name__ == "__main__":
    main()
