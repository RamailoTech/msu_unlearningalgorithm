# algorithms/saliency_unlearning/scripts/generate_mask.py

import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

# Ensure the parent directory is in the path
import sys
sys.path.append('../../..')  # Adjust the path as necessary based on your project structure

from algorithms.saliency_unlearning.data_handler import SaliencyUnlearnDataHandler
from algorithms.saliency_unlearning.model import SaliencyUnlearnModel
from constants.const import theme_available, class_available
from algorithms.saliency_unlearning.utils import load_model_from_config

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)
    plt.close()

def generate_forget_style_mask(prompt, output_dir, forget_data_dir, remain_data_dir, c_guidance, batch_size, config_path, ckpt_path, device, lr=1e-5, image_size=512, num_timesteps=1000, threshold=0.5):
    """
    Generate a saliency mask by training the model to forget specific styles.

    Args:
        prompt (str): The prompt describing the style to forget.
        output_dir (str): Directory to save the generated mask.
        forget_data_dir (str): Directory containing forget data.
        remain_data_dir (str): Directory containing remain data.
        c_guidance (float): Guidance scale for conditioning.
        batch_size (int): Batch size for training.
        config_path (str): Path to the model configuration file.
        ckpt_path (str): Path to the model checkpoint.
        device (str): Device to perform training on.
        lr (float, optional): Learning rate for optimizer. Defaults to 1e-5.
        image_size (int, optional): Size to resize images. Defaults to 512.
        num_timesteps (int, optional): Number of timesteps for diffusion. Defaults to 1000.
        threshold (float, optional): Threshold for mask generation. Defaults to 0.5.
    """
    # Initialize Data Handler
    data_handler = SaliencyUnlearnDataHandler(
        original_data_dir=forget_data_dir,
        new_data_dir=remain_data_dir,
        mask_path=None,  # Not needed for mask generation
        selected_theme='',
        selected_class='',
        use_sample=False,
        batch_size=batch_size,
        image_size=image_size,
        interpolation='bicubic',
        num_workers=4,
        pin_memory=True
    )

    # Initialize Model
    model = SaliencyUnlearnModel(
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask={},  # No mask applied during mask generation
        device=device
    )

    # Set model to train
    model.model.train()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)

    gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = 0.0

    # Initialize Logger
    logger = logging.getLogger('generate_mask')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # TRAINING CODE
    for epoch in range(1):  # Single epoch for mask generation
        logger.info(f"Starting Epoch {epoch+1}")
        forget_loader, _ = data_handler.get_data_loaders()['forget'], data_handler.get_data_loaders()['remain']

        with tqdm(total=len(forget_loader), desc=f'Epoch {epoch+1}/1') as t_bar:
            for i, forget_batch in enumerate(forget_loader):
                optimizer.zero_grad()

                images, _ = forget_batch  # Assuming dataset returns (images, prompts)
                images = images.to(device)
                t = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()

                prompts = [prompt] * images.size(0)

                forget_batch_dict = {
                    "edited": images,
                    "edit": {"c_crossattn": prompts}
                }

                null_batch_dict = {
                    "edited": images,
                    "edit": {"c_crossattn": [""] * images.size(0)}
                }

                forget_input, forget_emb = model.model.get_input(forget_batch_dict, model.model.first_stage_key)
                null_input, null_emb = model.model.get_input(null_batch_dict, model.model.first_stage_key)

                t = torch.randint(0, model.model.num_timesteps, (forget_input.shape[0],), device=device).long()
                noise = torch.randn_like(forget_input, device=device)

                forget_noisy = model.model.q_sample(x_start=forget_input, t=t, noise=noise)

                forget_out = model.model.apply_model(forget_noisy, t, forget_emb)
                null_out = model.model.apply_model(forget_noisy, t, null_emb)

                preds = (1 + c_guidance) * forget_out - c_guidance * null_out

                loss = -criteria(noise, preds)
                loss.backward()
                optimizer.step()

                # Accumulate gradients
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += torch.abs(param.grad.data.cpu())

                # Logging
                t_bar.set_postfix({"loss": loss.item() / batch_size})
                t_bar.update(1)

    # Aggregate and threshold gradients to create mask
    for name in gradients:
        gradients[name] = gradients[name].cpu()

    sorted_dict_positions = {}
    hard_dict = {}

    # Concatenate all tensors into a single tensor
    all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])

    # Calculate the threshold index for the top threshold% elements
    threshold_index = int(len(all_elements) * threshold)

    # Calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()

        tensor_ranks = ranks[start_index: start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor

        start_index += num_elements

    # Save the generated mask
    mask_save_path = os.path.join(output_dir, f'{threshold}.pt')
    torch.save(hard_dict, mask_save_path)
    logger.info(f"Mask saved at {mask_save_path}")

def main():
    parser = argparse.ArgumentParser(prog='GenerateMask', description='Generate saliency mask for saliency unlearning.')
    
    parser.add_argument('--c_guidance', help='Guidance scale used in loss computation', type=float, required=False, default=7.5)
    parser.add_argument('--batch_size', help='Batch size used for mask generation', type=int, required=False, default=4)
    parser.add_argument('--ckpt_path', help='Checkpoint path for the Stable Diffusion model', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True, help='Path to the model configuration file')
    parser.add_argument('--num_timesteps', help='Number of timesteps for diffusion', type=int, required=False, default=1000)
    parser.add_argument('--theme', help='Theme used for mask generation', type=str, required=True, choices=theme_available + class_available)
    parser.add_argument('--output_dir', help='Output directory for the generated mask', type=str, required=True)
    parser.add_argument('--threshold', help='Threshold for mask generation', type=float, required=False, default=0.5)
    parser.add_argument('--forget_data_dir', help='Directory containing forget data', type=str, required=True)
    parser.add_argument('--remain_data_dir', help='Directory containing remain data', type=str, required=True, default='data/Seed_Images/')
    parser.add_argument('--image_size', help='Image size for training', type=int, default=512)
    parser.add_argument('--lr', help='Learning rate for optimizer', type=float, default=1e-5)

    args = parser.parse_args()

    device = args.device if 'device' in args else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    prompt = f"An image in {args.theme} Style."
    output_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(output_dir, exist_ok=True)

    mask_save_path = os.path.join(output_dir, f'{args.threshold}.pt')
    if os.path.exists(mask_save_path):
        print(f"Mask for threshold {args.threshold} already exists. Skipping.")
        exit(0)

    forget_data_dir = os.path.join(args.forget_data_dir, args.theme)

    generate_forget_style_mask(
        prompt=prompt,
        output_dir=output_dir,
        forget_data_dir=forget_data_dir,
        remain_data_dir=args.remain_data_dir,
        c_guidance=args.c_guidance,
        batch_size=args.batch_size,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        device=device,
        lr=args.lr,
        image_size=args.image_size,
        num_timesteps=args.num_timesteps,
        threshold=args.threshold
    )

if __name__ == '__main__':
    main()
