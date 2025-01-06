from setuptools import setup, find_packages
import os

# Read the README file for the long description
with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Collect all environment.yaml files
def package_data_files():
    data_files = []
    for root, dirs, files in os.walk("mu/algorithms"):
        for file in files:
            if file == "environment.yaml":
                data_files.append(os.path.join(root, file))
    return data_files

setup(
    name="unlearn_diff",  # Replace with your package name
    version="0.0.1",
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
    ],
    extras_require={
    },
    entry_points={
        'console_scripts': [
            'create_env=mu.helpers.env_manager:main',
        ],
    },
)
