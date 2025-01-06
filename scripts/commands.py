import os
import subprocess
import sys
import argparse
import yaml

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def execute_script(script_path, *args):
    """Executes a bash script with arguments."""
    try:
        command = [script_path] + list(args)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error executing {script_path}: {e}")

def download_data(dataset_type, dataset_name):
    """Handles data download for i2p or quick_canvas."""
    if dataset_name not in ["i2p", "quick_canvas"]:
        sys.exit("Error: dataset_name must be 'i2p' or 'quick_canvas'.")
    
    if dataset_type not in ["full", "sample"]:
        sys.exit("Error: dataset_type must be 'full' or 'sample'.")

    script_map = {
        "i2p": os.path.join(BASE_DIR, "download_i2p_dataset.sh"),
        "quick_canvas": os.path.join(BASE_DIR, "download_quick_canvas_dataset.sh")
    }

    script_path = script_map[dataset_name]
    if not os.path.exists(script_path):
        sys.exit(f"Error: Script {script_path} not found.")

    execute_script(script_path, dataset_type)

def download_models(model_name):
    """Handles model downloads."""
    if model_name not in ["diffuser", "compvis"]:
        sys.exit("Error: model_name must be 'diffuser' or 'compvis'.")

    script_path = os.path.join(BASE_DIR, "download_models.sh")
    if not os.path.exists(script_path):
        sys.exit(f"Error: Script {script_path} not found.")

    execute_script(script_path, model_name)

def env_manager(algorithm, env_name=None):
    """Creates a Conda environment for a specific algorithm."""
    env_file = os.path.join(BASE_DIR, '..', 'mu', 'algorithms', algorithm, 'environment.yaml')

    if not os.path.exists(env_file):
        sys.exit(f"Environment file not found for algorithm '{algorithm}'.")
    with open(env_file, 'r') as file:
        env_data = yaml.safe_load(file)
        default_env_name = env_data.get('name', f"{algorithm}_env")

    env_name = env_name if env_name else default_env_name
    cmd = ["conda", "env", "create", "--name", env_name, "--file", env_file]
    try:
        subprocess.check_call(cmd)
        print(f"Conda environment '{env_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Failed to create Conda environment: {e}")

# Wrapper Functions for CLI

def download_data_cli():
    """CLI wrapper for download_data."""
    parser = argparse.ArgumentParser(description="Download datasets.")
    parser.add_argument(
        "dataset_type",
        choices=["full", "sample"],
        help="Type of dataset to download."
    )
    parser.add_argument(
        "dataset_name",
        choices=["i2p", "quick_canvas"],
        help="Name of the dataset to download."
    )
    args = parser.parse_args()
    download_data(args.dataset_type, args.dataset_name)

def download_models_cli():
    """CLI wrapper for download_models."""
    parser = argparse.ArgumentParser(description="Download models.")
    parser.add_argument(
        "model_name",
        choices=["diffuser", "compvis"],
        help="Name of the model to download."
    )
    args = parser.parse_args()
    download_models(args.model_name)

def create_env_cli():
    """CLI wrapper for env_manager."""
    parser = argparse.ArgumentParser(description="Create a Conda environment for an algorithm.")
    parser.add_argument(
        "algorithm",
        help="Name of the algorithm for which to create the environment."
    )
    parser.add_argument(
        "--env_name",
        default=None,
        help="Name of the Conda environment (optional)."
    )
    args = parser.parse_args()
    env_manager(args.algorithm, args.env_name)
