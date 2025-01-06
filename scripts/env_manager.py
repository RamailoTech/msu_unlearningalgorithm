import os
import sys
import subprocess
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Create a Conda environment for a specific algorithm.")
    parser.add_argument('algorithm', type=str, help='Name of the algorithm')
    parser.add_argument('--env-name', type=str, default=None, help='Name of the Conda environment')
    args = parser.parse_args()

    algorithm = args.algorithm

    # Construct the path to the environment.yaml
    package_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(package_dir, '..','mu', 'algorithms', algorithm, 'environment.yaml')

    if not os.path.exists(env_file):
        print(f"Environment file not found for algorithm '{algorithm}'.")
        sys.exit(1)

    # Load the environment.yaml to get the default environment name if not provided
    with open(env_file, 'r') as file:
        env_data = yaml.safe_load(file)
        default_env_name = env_data.get('name', f"{algorithm}_env")

    env_name = args.env_name if args.env_name else default_env_name

    # Create the environment
    cmd = ["conda", "env", "create", "--name", env_name, "--file", env_file]
    try:
        subprocess.check_call(cmd)
        print(f"Conda environment '{env_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create Conda environment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
