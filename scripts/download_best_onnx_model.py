import os
from huggingface_hub import hf_hub_download

def download_best_onnx(
    repo_id: str = "Palistha01/best_onnx",
    filename: str = "best.onnx",
    local_dir: str = "files"
) -> str:
    """
    Downloads the 'best.onnx' file from the specified Hugging Face repository 
    and saves it to the provided local directory.
    
    Parameters:
        repo_id (str): The Hugging Face repository identifier.
                       Default is "Palistha01/best_onnx".
        filename (str): The name of the file to download.
                        Default is "best.onnx".
        local_dir (str): The local directory where the file should be saved.
                         Default is "files".
    
    Returns:
        str: The local path to the downloaded file.
    
    Example:
        >>> model_path = download_best_onnx()
        >>> print(f"Model saved at: {model_path}")
    """
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the file from Hugging Face Hub into the specified directory
    local_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir=local_dir
    )
    print(f"Model downloaded and saved to: {local_path}")
    return local_path




def main():
    model_path = download_best_onnx()
    print(f"Model downloaded and saved to: {model_path}")

if __name__ == "__main__":
    main()