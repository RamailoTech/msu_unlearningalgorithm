from setuptools import setup, find_packages
import os
import subprocess
import sys

with open("Readme.md", "r", encoding="utf-8") as fh:
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
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Conda is installed.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.stderr.write("Error: Conda is not installed.\n")
        sys.exit(1)

check_conda()

setup(
    name="unlearn_diff",  
    version="1.0.1",
    author="nebulaanish",
    author_email="nebulaanish@gmail.com",
    description="Unlearning Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RamailoTech/msu_unlearningalgorithm", 
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['mu/algorithms/**/environment.yaml'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'pyyaml',
        'setuptools',
    ],
    extras_require={
    },
    entry_points={
        'console_scripts': [
            'create_env=scripts.commands:create_env_cli',
            'download_data=scripts.commands:download_data_cli',
            'download_model=scripts.commands:download_models_cli',
        ],
    },
)
