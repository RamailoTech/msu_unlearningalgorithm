# erase_diff/utils.py

from omegaconf import OmegaConf
import torch
from typing import Any
from pathlib import Path
from mu.stable_diffusion.ldm.util import instantiate_from_config
from mu.stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
import gc
import numpy as np
from timm.models.layers import trunc_normal_
import copy
import quadprog
from torch.nn import MSELoss

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


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
    Solves the GEM dual QP described in the paper given a proposed
    gradient "gradient", and a memory of task gradients "memories".
    Overwrites "gradient" with the final projected update.

    Args:
        gradient (torch.Tensor): Proposed gradient.
        memories (torch.Tensor): Task gradient memory.
        margin (float): Margin constraint for projection.
        eps (float): Small value to stabilize QP solver.

    Returns:
        torch.Tensor: Projected gradient.
    """
    memories_np = memories.cpu().t().contiguous().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()

    t = memories_np.shape[0]  # Number of tasks
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]  # Solve the QP
    x = np.dot(v, memories_np) + gradient_np  # Compute the projected gradient
    new_grad = torch.Tensor(x).view(-1)
    return new_grad


def create_dense_mask(net, device, value=1):
    """
    Create a dense mask where all parameters are set to a specific value.

    Args:
        net: Model to apply the mask.
        device (str): Device to use.
        value (int): Value to set in the mask.

    Returns:
        net: Masked model.
    """
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


def snip(model, dataloader, sparsity, prune_num, device):
    """
    Apply SNIP-based pruning to the model.

    Args:
        model: Model to prune.
        dataloader: DataLoader for computing gradients.
        sparsity (float): Desired sparsity level.
        prune_num (int): Number of iterations to compute gradients.
        device (str): Device to use for computation.

    Returns:
        model: Pruned model.
    """
    grads = [torch.zeros_like(p) for p in model.model.model.diffusion_model.parameters()]
    criterion = MSELoss()

    # Compute gradients over multiple iterations
    for _ in range(prune_num):
        forget_images, forget_prompts = next(iter(dataloader))
        forget_prompts = list(forget_prompts)  # Convert tuple to list

        forget_batch = {
            "edited": forget_images.to(device),
            "edit": {"c_crossattn": forget_prompts}
        }
        loss = model.model.shared_step(forget_batch)[0]
        model.model.model.diffusion_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for i, param in enumerate(model.model.model.diffusion_model.parameters()):
                if param.grad is not None:
                    grads[i] += param.grad.abs()
            torch.cuda.empty_cache()
            gc.collect()

    # Compute saliency scores
    weights = [p for p in model.model.model.diffusion_model.parameters()]
    mask = create_dense_mask(copy.deepcopy(model.model.model.diffusion_model), device, value=1)

    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for grad, weight in zip(grads, weights)]
        flat_saliences = torch.cat([s.view(-1).cpu() for s in abs_saliences])
        threshold = float(flat_saliences.kthvalue(int(sparsity * flat_saliences.numel()))[0])

        # Prune weights based on the threshold
        for i, param in enumerate(mask.parameters()):
            indices = abs_saliences[i] > threshold
            param.data[indices] = 0

        # Update the model parameters with the mask
        for (name, param), mask_param in zip(model.model.model.diffusion_model.named_parameters(), mask.parameters()):
            if "attn2" in name:
                mask_tensor = torch.empty_like(param.data)
                if "weight" in name:
                    re_init_param = trunc_normal_(mask_tensor, std=0.02)
                elif "bias" in name:
                    re_init_param = torch.nn.init.zeros_(mask_tensor)
                param.data = param.data * mask_param.data + re_init_param.data * (1 - mask_param.data)

    return model
