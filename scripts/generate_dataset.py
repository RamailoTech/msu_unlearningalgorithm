import argparse
import os
from mu_attack.datasets.dataset_generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of images using Diffusers.")
    parser.add_argument('--prompts_path', type=str, required=True, help="Path to the CSV file containing prompts.")
    parser.add_argument('--save_path', type=str, required=True, help="Directory where generated images and metadata will be saved.")
    parser.add_argument('--concept', type=str, default='default', help="Name of the concept for organizing output files.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the model on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale for classifier-free diffusion guidance.")
    parser.add_argument('--image_size', type=int, default=512, help="Size of the generated images (e.g., 512).")
    parser.add_argument('--ddim_steps', type=int, default=100, help="Number of diffusion steps (e.g., 100).")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of images to generate per prompt.")
    parser.add_argument('--from_case', type=int, default=0, help="Start processing from this case number (e.g., 0).")
    parser.add_argument('--cache_dir', type=str, default='./ldm_pretrained', help="Directory to cache pre-trained model weights.")
    parser.add_argument('--ckpt', type=str, default=None, help="Path to a custom model checkpoint (optional).")

    args = parser.parse_args()

    # Initialize the generator with specified arguments
    generator = DatasetGenerator(
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir
    )

    # Generate images using the provided arguments
    generator.generate_images(
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        concept=args.concept,
        from_case=args.from_case,
        ckpt=args.ckpt
    )

if __name__ == "__main__":
    main()
