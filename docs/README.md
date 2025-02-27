## Usage
This section contains the usage guide for the package.

### Installation

#### Prerequisities
Ensure `conda` is installed on your system. You can install Miniconda or Anaconda:

- **Miniconda** (recommended): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Anaconda**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

After installing `conda`, ensure it is available in your PATH by running. You may require to restart the terminal session:

Before installing the unlearn_diff package, follow these steps to set up your environment correctly. These instructions ensure compatibility with the required dependencies, including Python, PyTorch, and ONNX Runtime.


**Step-by-Step Setup:**

Step 1. Create a Conda Environment Create a new Conda environment named myenv with Python 3.8.5:

```bash
conda create -n myenv python=3.8.5
```

Step 2. Activate the Environment Activate the environment to work within it:

```bash
conda activate myenv
```

Step 3. Install Core Dependencies Install PyTorch, torchvision, CUDA Toolkit, and ONNX Runtime with specific versions:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 onnxruntime==1.16.3 -c pytorch -c conda-forge
```

Step 4. Install our unlearn_diff Package using pip:

```bash
pip install unlearn_diff
```

Step 5. Install Additional Git Dependencies:

 After installing unlearn_diff, install the following Git-based dependencies in the same Conda environment to ensure full functionality:

```bash
pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```

```bash
pip install git+https://github.com/openai/CLIP.git@main#egg=clip
```

```bash
pip install git+https://github.com/crowsonkb/k-diffusion.git
```

```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

```bash
pip install git+https://github.com/Phoveran/fastargs.git@main#egg=fastargs
```

```bash
pip install git+https://github.com/boomb0om/text2image-benchmark
```



### Downloading data and models.
After you install the package, you can use the following commands to download.

1. **Dataset**:
  - **i2p**:
    - **Sample**:
     ```
     download_data sample i2p
     ```
    - **Full**:
     ```
     download_data full i2p
     ```
  - **quick_canvas**:
    - **Sample**:
     ```
     download_data sample quick_canvas
     ```
    - **Full**:
     ```
     download_data full quick_canvas
     ```

2. **Model**:
  - **compvis**:
    ```
    download_model compvis
    ```
  - **diffuser**:
    ```
    download_model diffuser
    ```


### Run Train <br>

Each algorithm has their own script to run the algorithm, Some also have different process all together. Follow usage section in readme for the algorithm you want to run with the help of the github repository. You will need to run the code snippet provided in usage section with necessary configuration passed. 


### Example Usage: Running Machine Unlearning with the ESD Algorithm:

After successfully installing the package, you can run machine unlearning using the ESDAlgorithm. Below is a refined example that demonstrates how to import the necessary modules, configure the algorithm, and execute the unlearning process:

```python
    from mu.algorithms.esd.algorithm import ESDAlgorithm
    from mu.algorithms.esd.configs import esd_train_mu

    algorithm = ESDAlgorithm(
        esd_train_mu,
        ckpt_path="models/compvis/style50/compvis.ckpt",
        raw_dataset_dir="data/quick-canvas-dataset/sample",
        use_sample = True, #uses sample dataset
        template_name = "Abstractionism",
        dataset_type = "unlearncanvas"
    )
    algorithm.run()
```

For detailed information on the ESD algorithm and its configuration, please refer to the detailed documentation [here](https://ramailotech.github.io/msu_unlearningalgorithm/unlearn/algorithms/esd/).

Refer [here](https://ramailotech.github.io/msu_unlearningalgorithm/unlearn/examples/esd/#) for instructions on overriding default configurations and creating your own configuration class. You can also review detailed configuration descriptions [here](https://ramailotech.github.io/msu_unlearningalgorithm/unlearn/configs/esd/)


**Link to our example usage notebooks**

Additionally, we have provided comprehensive notebooks that cover the complete workflow—from unlearning to attack and defense—along with their evaluation. Please refer to the notebooks below for full example usage.

1. **Erase-diff (compvis model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_erase_diff.ipynb

2. **forget-me-not (Diffuser model)**

https://github.com/RamailoTech/msu_unlearningalgorithm/blob/main/notebooks/run_forget_me_not.ipynb
