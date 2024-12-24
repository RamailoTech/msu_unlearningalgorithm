# erase_diff/utils.py

from omegaconf import OmegaConf
import torch
from typing import Any
from pathlib import Path
from mu.stable_diffusion.ldm.util import instantiate_from_config
from mu.stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config_path: str, ckpt_path: str, device: str = "cpu") -> Any:
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
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, num_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
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
        till_T=till_T
    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim
