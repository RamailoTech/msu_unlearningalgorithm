# from setuptools import setup, find_packages
# import os
# import subprocess
# import sys

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

# def package_data_files():
#     data_files = []
#     for root, dirs, files in os.walk("mu/algorithms"):
#         for file in files:
#             if file == "environment.yaml":
#                 data_files.append(os.path.join(root, file))
#     return data_files

# def check_conda():
#     try:
#         subprocess.run(
#             ["conda", "--version"],
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
#         print("Conda is installed.")
#     except (FileNotFoundError, subprocess.CalledProcessError):
#         sys.stderr.write("Error: Conda is not installed.\n")
#         sys.exit(1)

# check_conda()

# def parse_requirements(filename):
#     """Load dependencies from a pip requirements file."""
#     with open(filename, "r") as req_file:
#         return [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

# setup(
#     name="unlearn_diff",
#     version="1.0.4",
#     author="nebulaanish",
#     author_email="nebulaanish@gmail.com",
#     description="Unlearning Algorithms",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/RamailoTech/msu_unlearningalgorithm",
#     project_urls={
#         "Documentation": "https://ramailotech.github.io/msu_unlearningalgorithm/",
#         "Source Code": "https://github.com/RamailoTech/msu_unlearningalgorithm",
#     },
#     packages=find_packages(),
#     include_package_data=True,  # Ensure this line is added only once
#     package_data={
#         "": ["mu/algorithms/**/environment.yaml"],
#     },
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     python_requires=">=3.8, <3.9",  # Set the Python version requirement here
#     install_requires=parse_requirements("requirements.txt") ,  # Install dependencies from requirements.txt
#     extras_require={},
#     entry_points={
#         "console_scripts": [
#             "create_env=scripts.commands:create_env_cli",
#             "download_data=scripts.commands:download_data_cli",
#             "download_model=scripts.commands:download_models_cli",
#         ],
#     },
# )


from setuptools import setup, find_packages
import os
import subprocess
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def package_data_files():
    data_files = []
    for root, dirs, files in os.walk("mu/algorithms"):
        for file in files:
            if file == "environment.yaml":
                data_files.append(os.path.join(root, file))
    return data_files

def check_conda():
    try:
        subprocess.run(
            ["conda", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Conda is installed.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.stderr.write("Error: Conda is not installed.\n")
        sys.exit(1)

def create_and_activate_conda_env():
    """Creates or updates a conda environment using the provided environment.yaml file and activates it."""
    yaml_file = "environment.yaml"
    if not os.path.exists(yaml_file):
        sys.stderr.write(f"Error: {yaml_file} not found. Cannot create conda environment.\n")
        sys.exit(1)

    try:
        # Read the environment name from the YAML file
        env_name = None
        with open(yaml_file, "r") as f:
            for line in f:
                if line.startswith("name:"):
                    env_name = line.split(":")[1].strip()
                    break

        if not env_name:
            sys.stderr.write(f"Error: Could not find the 'name' field in {yaml_file}.\n")
            sys.exit(1)

        # Check if the environment already exists
        result = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if env_name in result.stdout:
            print(f"Updating the Conda environment '{env_name}' from {yaml_file}...")
            subprocess.run(
                ["conda", "env", "update", "-f", yaml_file],
                check=True,
            )
        else:
            print(f"Creating the Conda environment '{env_name}' from {yaml_file}...")
            subprocess.run(
                ["conda", "env", "create", "-f", yaml_file],
                check=True,
            )

        # Temporarily activate the environment
        print(f"Activating the Conda environment '{env_name}'...")
        activate_command = f"conda activate {env_name} && echo 'Environment {env_name} activated.'"
        subprocess.run(activate_command, shell=True, executable="/bin/bash")

        print(f"Conda environment '{env_name}' is ready. Please activate it manually: 'conda activate {env_name}'.")

    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error creating or updating conda environment: {e.stderr}\n")
        sys.exit(1)

# Check if conda is installed
check_conda()

# Create the conda environment
create_and_activate_conda_env()

setup(
    name="unlearn_diff",
    version="1.0.4",
    author="nebulaanish",
    author_email="nebulaanish@gmail.com",
    description="Unlearning Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RamailoTech/msu_unlearningalgorithm",
    project_urls={
        "Documentation": "https://ramailotech.github.io/msu_unlearningalgorithm/",
        "Source Code": "https://github.com/RamailoTech/msu_unlearningalgorithm",
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["mu/algorithms/**/environment.yaml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pyyaml",
        "setuptools",
    ],
    extras_require={},
    entry_points={
        "console_scripts": [
            "create_env=scripts.commands:create_env_cli",
            "download_data=scripts.commands:download_data_cli",
            "download_model=scripts.commands:download_models_cli",
        ],
    },
)
