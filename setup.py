from setuptools import setup, find_packages
from setuptools.command.install import install as _install
import os
import subprocess
import sys
import json
import yaml

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


check_conda()


class CustomInstallCommand(_install):
    def run(self):
        # Run the standard installation process.
        _install.run(self)

        # Step 1: Use the environment.yaml to either update or create the conda environment.
        print("\nProcessing environment.yaml ...")
        env_yaml = os.path.join(os.path.dirname(__file__), "environment.yaml")
        if not os.path.exists(env_yaml):
            sys.stderr.write(f"Error: {env_yaml} not found.\n")
            sys.exit(1)

        # Parse environment.yaml to get the environment name.
        try:
            with open(env_yaml, "r", encoding="utf-8") as f:
                env_data = yaml.safe_load(f)
            env_name = env_data.get("name")
            if not env_name:
                sys.stderr.write(
                    "Error: Environment name not found in environment.yaml.\n"
                )
                sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error reading {env_yaml}: {e}\n")
            sys.exit(1)

        # Check if the environment exists by listing current conda environments.
        try:
            output = subprocess.check_output(
                ["conda", "env", "list", "--json"], universal_newlines=True
            )
            envs = json.loads(output).get("envs", [])
            # Compare basenames of the environment paths to the desired env_name.
            env_exists = any(
                os.path.basename(env_path) == env_name for env_path in envs
            )
        except Exception as e:
            sys.stderr.write(f"Error checking conda environments: {e}\n")
            sys.exit(1)

        if env_exists:
            print(f"Environment '{env_name}' exists.")
        else:
            print(f"Environment '{env_name}' does not exist. Creating environment...")
            try:
                subprocess.check_call(
                    ["conda", "env", "create", "--name", env_name, "--file", env_yaml]
                )
                print("Conda environment created successfully.")
            except subprocess.CalledProcessError as e:
                sys.stderr.write(f"Error creating conda environment: {e}\n")
                sys.exit(1)


setup(
    name="unlearn_diff",
    version="1.0.5",
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
        "License :: OSI Approved :: MIT License",  # Choose appropriate license
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
    cmdclass={
        "install": CustomInstallCommand,
    },
)
