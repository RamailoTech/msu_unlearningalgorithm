import os
import argparse
import shutil
import random
import pandas as pd
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_images(model_path, csv_path):
    # Derive the base directory name from the CSV filename (without extension)
    csv_filename = os.path.basename(csv_path)
    csv_basename, _ = os.path.splitext(csv_filename)
    base_dir = os.path.join("data", csv_basename)
    
    # Create directories for images and prompts inside the base directory.
    images_base_dir = os.path.join(base_dir, "images")
    prompts_base_dir = os.path.join(base_dir, "prompts")
    os.makedirs(images_base_dir, exist_ok=True)
    os.makedirs(prompts_base_dir, exist_ok=True)
    
    # Copy the input CSV file into the prompts folder.
    copied_csv_path = os.path.join(prompts_base_dir, csv_filename)
    shutil.copy(csv_path, copied_csv_path)
    print(f"Copied CSV to: {copied_csv_path}")
    
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Verify required columns exist in the CSV (only "prompt" and "categories" are required)
    required_columns = {"prompt", "categories"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {', '.join(missing_columns)}")
    
    # Determine if 'case_number' exists; otherwise, use auto-increment counters.
    use_auto_increment = "case_number" not in data.columns
    auto_counters = {}  # Counters per category

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe.to("cuda")

    def sanitize_category(category):
        """Sanitize category string to create valid folder names."""
        return category.replace(",", "_").replace(" ", "_").lower()

    # Iterate through each row in the CSV file.
    for index, row in data.iterrows():
        prompt = row["prompt"]
        # Split multiple categories (assuming they are comma-separated)
        categories = [cat.strip() for cat in row["categories"].split(",")]
        
        # Determine filename using case_number if available, or auto-increment.
        if not use_auto_increment:
            case_number = row.get("case_number")
            if pd.isna(case_number) or str(case_number).strip() == "":
                use_auto = True
            else:
                use_auto = False
        else:
            use_auto = True

        for category in categories:
            sanitized_category = sanitize_category(category)
            # Create a subdirectory for the category within the images folder.
            category_image_dir = os.path.join(images_base_dir, sanitized_category)
            os.makedirs(category_image_dir, exist_ok=True)
            
            # Determine filename for the image.
            if use_auto:
                if sanitized_category not in auto_counters:
                    auto_counters[sanitized_category] = 0
                file_number = auto_counters[sanitized_category]
                image_filename = f"{file_number}.jpg"
                auto_counters[sanitized_category] += 1
            else:
                image_filename = f"{row['case_number']}.jpg"
            
            image_output_path = os.path.join(category_image_dir, image_filename)
            if os.path.exists(image_output_path):
                print(f"Image already exists for case {image_filename} in category {sanitized_category}. Skipping.")
                continue

            print(f"Generating image for case {image_filename} in category {sanitized_category}...")
            try:
                with autocast("cuda"):
                    image = pipe(prompt).images[0]
                image.save(image_output_path)
                print(f"Saved image: {image_output_path}")
            except Exception as e:
                print(f"Failed to generate image for case {image_filename}: {e}")

    print("Image generation completed.")

    # Create a Seed_Images folder inside images folder.
    seed_dir = os.path.join(images_base_dir, "Seed_Images")
    os.makedirs(seed_dir, exist_ok=True)

    # Generate images for the top 5 prompts
    top_prompts = data["prompt"].head(2)  # Select top 5 prompts from the CSV (you can adjust this if needed) #TODO chnage the number for seed images

    seed_counter = 0
    for prompt in top_prompts:
        print(f"Generating seed image for prompt: {prompt}...")
        try:
            with autocast("cuda"):
                image = pipe(prompt).images[0]
            seed_image_path = os.path.join(seed_dir, f"{seed_counter}.jpg")
            image.save(seed_image_path)
            print(f"Saved seed image: {seed_image_path}")
            seed_counter += 1
        except Exception as e:
            print(f"Failed to generate seed image for prompt: {prompt}. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--model_path", required=True, help="Path to the model.")
    parser.add_argument("--csv_path", required=True, help="Path to the CSV file with prompts.")
    
    args = parser.parse_args()
    generate_images(args.model_path, args.csv_path)

if __name__ == "__main__":
    main()
