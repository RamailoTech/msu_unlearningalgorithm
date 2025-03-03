# scripts/generate_images_for_prompts.py

import os
import argparse
import pandas as pd
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_images(model_path, csv_path):
    # Derive base_dir from the CSV filename (without extension)
    csv_filename = os.path.basename(csv_path)
    csv_basename, _ = os.path.splitext(csv_filename)
    base_dir = os.path.join("data", csv_basename)
    
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Verify required columns exist in the CSV
    required_columns = {"prompt", "categories", "case_number"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {', '.join(missing_columns)}")

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe.to("cuda")

    # Create the base directory for saving images
    os.makedirs(base_dir, exist_ok=True)

    def sanitize_category(category):
        """Sanitize category string to create valid folder names."""
        return category.replace(",", "_").replace(" ", "_")

    # Iterate through each row in the CSV
    for index, row in data.iterrows():
        prompt = row["prompt"]
        categories = row["categories"].split(", ")  # Split multiple categories
        case_number = row["case_number"]

        for category in categories:
            sanitized_category = sanitize_category(category)
            output_dir = os.path.join(base_dir, sanitized_category)
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{case_number}.jpg")

            if os.path.exists(output_path):
                print(f"Image already exists for case {case_number} in category {sanitized_category}. Skipping.")
                continue

            print(f"Generating image for case {case_number} in category {sanitized_category}...")
            try:
                with autocast("cuda"):
                    image = pipe(prompt).images[0]
                image.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed to generate image for case {case_number}: {e}")

    print("Image generation completed.")

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--model_path", required=True, help="Path to the model.")
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file with prompts.")
    
    args = parser.parse_args()
    generate_images(args.model_path, args.csv_path)
