import os
import pandas as pd
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the CSV file
csv_path = "/home/ubuntu/Projects/msu_unlearningalgorithm/data/i2p-dataset/sample/i2p.csv"
data = pd.read_csv(csv_path)

# Stable Diffusion model path
model_path = "/home/ubuntu/Projects/UnlearnCanvas/UnlearnCanvas/machine_unlearning/models/diffuser/style50"  # Replace with your model path
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe.to("cuda")

# Base directory for saving images
base_dir = "/home/ubuntu/Projects/msu_unlearningalgorithm/data/i2p-dataset/sample"
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
