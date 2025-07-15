import os
import gdown
import zipfile
from pathlib import Path

def download_reference_images(
    gdrive_id: str = '1Qgm3nNhp6ykamszN_ZvofvuzjryTsPHB',
    output_dir: str = './data/coco_reference_images'
):
    """
    Downloads and extracts reference images from a Google Drive link.

    Args:
        gdrive_id (str): The file ID from the Google Drive sharing link.
        output_dir (str): The directory where the images will be saved.

    Returns:
        str: The path to the directory containing the extracted images.
    """
    output_path = Path(output_dir)
    
    # Check if the directory already exists and has content
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Reference images already exist in '{output_dir}'. Skipping download.")
        return str(output_path)

    print(f"Downloading reference images to '{output_dir}'...")
    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / 'coco_reference.zip'

    try:
        # Download the file from Google Drive
        gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)

        # Unzip the file
        print(f"Extracting files to '{output_dir}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        # Clean up the downloaded zip file
        os.remove(zip_path)
        print("Download and extraction complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up partial downloads or corrupted files
        if zip_path.exists():
            os.remove(zip_path)
        return None

    return str(output_path)


def main():
    reference_image_path = download_reference_images()

    if reference_image_path:
        print(f"\nCOCO Images are ready at: {reference_image_path}")
        
if __name__ == '__main__':
    main()

