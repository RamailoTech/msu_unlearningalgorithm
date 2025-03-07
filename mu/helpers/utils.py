import sys
import os
import glob
from typing import List, Any
import argparse
from omegaconf import OmegaConf
import torch

from pytorch_lightning.utilities.distributed import rank_zero_only
from pathlib import Path
import pandas as pd

from models import stable_diffusion  
sys.modules['stable_diffusion'] = stable_diffusion


from stable_diffusion.ldm.util import instantiate_from_config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_text_lines(path: str) -> List[str]:
    """Read lines from a text file and strip whitespace."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def load_model_from_config(
    config_path: str, ckpt_path: str, device: str = "cpu"
) -> Any:
    """
    Load a model from a config file and checkpoint.

    Args:
        config_path (str): Path to the model configuration file.
        ckpt_path (str): Path to the model checkpoint.
        device (str, optional): Device to load the model on. Defaults to "cpu".

    Returns:
        Any: Loaded model.
    """
    if isinstance(config_path, (str, Path)):
        config = OmegaConf.load(config_path)
    else:
        config = config_path  # If already a config object

    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(
    model,   
    sampler,
    c,
    h,
    w,
    ddim_steps,
    scale,
    ddim_eta,
    start_code=None,
    num_samples=1,
    t_start=-1,
    log_every_t=None,
    till_T=None,
    verbose=True,
):
    """
    Generate samples using the sampler.

    Args:
        model (torch.nn.Module): The Stable Diffusion model.
        sampler (DDIMSampler): The sampler instance.
        c (Any): Conditioning tensors.
        h (int): Height of the image.
        w (int): Width of the image.
        ddim_steps (int): Number of DDIM steps.
        scale (float): Unconditional guidance scale.
        ddim_eta (float): DDIM eta parameter.
        start_code (torch.Tensor, optional): Starting latent code. Defaults to None.
        num_samples (int, optional): Number of samples to generate. Defaults to 1.
        t_start (int, optional): Starting timestep. Defaults to -1.
        log_every_t (int, optional): Logging interval. Defaults to None.
        till_T (int, optional): Timestep to stop sampling. Defaults to None.
        verbose (bool, optional): Verbosity flag. Defaults to True.

    Returns:
        torch.Tensor: Generated samples.
    """
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(num_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=num_samples,
        shape=shape,
        verbose=False,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        t_start=t_start,
        log_every_t=log_t,
        till_T=till_T,
    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


def safe_dir(dir):
    """
    Create a directory if it does not exist.
    """
    if not dir.exists():
        dir.mkdir()
    return dir


def load_config_from_yaml(config_path):
    """
    Load a configuration from a YAML file.
    """
    if isinstance(config_path, (str, Path)):
        config = OmegaConf.load(config_path)
    else:
        config = config_path  # If already a config object

    return config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def load_ckpt_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config["model"])
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def to_cuda(elements):
    """Transfers elements to CUDA if GPU is available."""
    if torch.cuda.is_available():
        return elements.to("cuda")
    return elements


def load_categories(reference_dir: str) -> list:
    """
    Load unique categories from a CSV file in the 'prompts' folder under the given reference directory.
    
    Args:
        reference_dir (str): The base directory where the 'prompts' folder is located.
        
    Returns:
        List[str]: A sorted list of unique categories extracted from the CSV file.
        
    Raises:
        FileNotFoundError: If no CSV file is found in the prompts folder or if the file doesn't exist.
    """
    prompts_folder = os.path.join(reference_dir, 'prompts')
    prompts_files = glob.glob(os.path.join(prompts_folder, '*.csv'))
    if not prompts_files:
        raise FileNotFoundError(f"No CSV file found in the prompts folder: {prompts_folder}")
    
    prompts_file = prompts_files[0]
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    data = pd.read_csv(prompts_file)
    unique_categories = set()
    for cats in data['categories']:
        if isinstance(cats, str):
            for cat in cats.split(','):
                unique_categories.add(cat.strip())
    categories = sorted(list(unique_categories))
    return categories


