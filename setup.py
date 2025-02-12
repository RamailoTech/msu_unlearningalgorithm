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

# You can keep this check if your package requires a conda environment.
check_conda()

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
    # Only include packages that can be installed via pip
    install_requires=[
        "albumentations==0.4.3",
        "datasets==2.8.0",
        "opencv-python==4.1.2.30",
        "pudb==2019.2",
        "invisible-watermark",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.2",
        "pytorch-lightning==1.4.2",
        "omegaconf==2.1.1",
        "test-tube>=0.7.5",
        "streamlit>=0.73.1",
        "einops==0.3.0",
        "torch-fidelity==0.3.0",
        "transformers==4.36.0",
        "torchmetrics==0.6.0",
        "kornia==0.6",
        "taming-transformers-rom1504",  # assuming available on PyPI; otherwise use dependency_links or document manual install
        "clip",  # same note as above
        "openai",
        "gradio",
        "loguru",
        "ml_collections",
        "webdataset",
        "ftfy",
        "yacs",
        "controlnet_aux",
        "fvcore",
        "h5py",
        "xtcocotools",
        "natsort",
        "timm==0.6.7",
        "fairscale",
        "open_clip_torch",
    ],
    # For packages installed from git repositories, you can use dependency_links (though note that pip support is diminishing)
    dependency_links=[
        "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
        "git+https://github.com/openai/CLIP.git@main#egg=clip",
        "git+https://github.com/crowsonkb/k-diffusion.git",
        "git+https://github.com/cocodataset/panopticapi.git",
        "git+https://github.com/facebookresearch/detectron2.git",
    ],
    entry_points={
        "console_scripts": [
            "create_env=scripts.commands:create_env_cli",
            "download_data=scripts.commands:download_data_cli",
            "download_model=scripts.commands:download_models_cli",
        ],
    },
)