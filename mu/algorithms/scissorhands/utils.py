# mu/algorithms/scissorhands/utils.py

import torch
from pathlib import Path
import gc
import numpy as np
from timm.models.layers import trunc_normal_
import copy
import quadprog
from torch.nn import MSELoss


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
    grads = [torch.zeros_like(p) for p in model.model.diffusion_model.parameters()]
    criterion = MSELoss()

    # Compute gradients over multiple iterations
    for _ in range(prune_num):
        forget_images, forget_prompts = next(iter(dataloader))
        forget_prompts = list(forget_prompts)  # Convert tuple to list

        forget_batch = {
            "edited": forget_images.to(device),
            "edit": {"c_crossattn": forget_prompts}
        }
        loss = model.shared_step(forget_batch)[0]
        model.model.diffusion_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            j = 0
            for n, param in  model.model.diffusion_model.named_parameters():
                if (param.grad is not None):
                    grads[j] += (param.grad.data).abs()
                j += 1
            torch.cuda.empty_cache()
            gc.collect()

    # Compute saliency scores
    weights = [p for p in model.model.diffusion_model.parameters()]
    mask = create_dense_mask(copy.deepcopy(model.model.diffusion_model), device, value=1)

    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for grad, weight in zip(grads, weights)]
        flat_saliences = torch.cat([s.view(-1).cpu() for s in abs_saliences])
        threshold = float(flat_saliences.kthvalue(int(sparsity * flat_saliences.numel()))[0])

        # Prune weights based on the threshold
        for i, param in enumerate(mask.parameters()):
            indices = abs_saliences[i] > threshold
            param.data[indices] = 0

        # Update the model parameters with the mask
        for (name, param), mask_param in zip(model.model.diffusion_model.named_parameters(), mask.parameters()):
            if "attn2" in name:
                mask_tensor = torch.empty_like(param.data)
                if "weight" in name:
                    re_init_param = trunc_normal_(mask_tensor, std=0.02)
                elif "bias" in name:
                    re_init_param = torch.nn.init.zeros_(mask_tensor)
                param.data = param.data * mask_param.data + re_init_param.data * (1 - mask_param.data)

    return model
