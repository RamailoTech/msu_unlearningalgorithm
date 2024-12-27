import torch
from torch.nn import MSELoss
from tqdm import tqdm
import os
import logging

def accumulate_gradients_for_mask(model, forget_loader, prompt, c_guidance, device, lr=1e-5, num_timesteps=1000, threshold=0.5, batch_size=4):
    """
    Run a single pass of gradient accumulation over the forget_loader to generate a saliency mask.
    This function:
    - Initializes an optimizer and a loss (MSE).
    - Loops over the forget_loader for one epoch-like run.
    - Accumulates absolute gradients for each parameter.
    - After accumulation, generates a binary mask based on the threshold.
    """

    # Set model to train mode
    model.model.train()
    criteria = MSELoss()
    optimizer = torch.optim.Adam(model.model.model.diffusion_model.parameters(), lr=lr)

    # Dictionary to accumulate gradients
    gradients = {name: torch.zeros_like(param, device='cpu') for name, param in model.model.model.diffusion_model.named_parameters()}

    logger = logging.getLogger(__name__)
    logger.info("Starting gradient accumulation for mask generation...")

    # Single epoch-like pass
    with tqdm(total=len(forget_loader), desc='Accumulating Gradients for Mask') as pbar:
        for images, _ in forget_loader:
            optimizer.zero_grad()

            images = images.to(device)
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()

            prompts = [prompt] * images.size(0)
            null_prompts = [""] * images.size(0)

            # Prepare batches
            forget_batch = {
                "edited": images,
                "edit": {"c_crossattn": prompts}
            }
            null_batch = {
                "edited": images,
                "edit": {"c_crossattn": null_prompts}
            }

            forget_input, forget_emb = model.model.get_input(forget_batch, model.model.first_stage_key)
            null_input, null_emb = model.model.get_input(null_batch, model.model.first_stage_key)

            t = torch.randint(0, model.model.num_timesteps, (forget_input.shape[0],), device=device).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = model.model.q_sample(x_start=forget_input, t=t, noise=noise)

            forget_out = model.model.apply_model(forget_noisy, t, forget_emb)
            null_out = model.model.apply_model(forget_noisy, t, null_emb)

            preds = (1 + c_guidance) * forget_out - c_guidance * null_out

            loss = -criteria(noise, preds)
            loss.backward()
            optimizer.step()

            # Accumulate absolute gradients
            for name, param in model.model.model.diffusion_model.named_parameters():
                if param.grad is not None:
                    gradients[name] += torch.abs(param.grad.data.cpu())

            pbar.set_postfix({"loss": loss.item() / batch_size})
            pbar.update(1)

    # Now compute the mask based on threshold
    mask = create_mask_from_gradients_optimized(gradients, threshold)
    return mask

def create_mask_from_gradients_optimized(gradients, threshold):
    """
    Given a dictionary of gradients (accumulated absolute gradients) and a threshold,
    create a binary mask dictionary in a memory-efficient way.
    """
    # Compute the global threshold value without concatenating
    all_abs_values = torch.cat([torch.abs(g).flatten() for g in gradients.values()])
    threshold_value = torch.topk(all_abs_values, int(len(all_abs_values) * threshold)).values.min()

    # Create binary masks without full concatenation
    mask_dict = {}
    for key, tensor in gradients.items():
        mask_dict[key] = (torch.abs(tensor) >= threshold_value).to(tensor.dtype)

    return mask_dict

def create_mask_from_gradients(gradients, threshold):
    """
    Given a dictionary of gradients (accumulated absolute gradients) and a threshold,
    create a binary mask dictionary.
    """
    all_elements = torch.cat([g.flatten() for g in gradients.values()])
    # threshold_index for the top threshold% elements
    threshold_index = int(len(all_elements) * threshold)

    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    mask_dict = {}
    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()
        tensor_ranks = ranks[start_index:start_index + num_elements]

        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        mask_dict[key] = threshold_tensor

        start_index += num_elements

    return mask_dict


def save_mask(mask_dict, output_path):
    """
    Save the mask dictionary to a .pt file.
    """
    torch.save(mask_dict, output_path)
