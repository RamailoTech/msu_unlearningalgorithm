# utils.py

from omegaconf import OmegaConf
import torch
from typing import Any
from pathlib import Path

from latent_diffusion.ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, device="cpu"):
    """Loads a model from config and a checkpoint."""
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
    """Sample the model."""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    shape = [4, h // 8, w // 8]
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter=verbose,
                                     t_start=t_start,
                                     log_every_t=log_every_t,
                                     till_T=till_T
                                     )
    return samples_ddim
