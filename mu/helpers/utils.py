from typing import List,Any
import argparse


from omegaconf import OmegaConf
import torch
# from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.distributed import rank_zero_only
from pathlib import Path
import multiprocessing

from stable_diffusion.ldm.util import instantiate_from_config

import torch
import cv2
import numpy as np
from torchvision.models import inception_v3
from torch import nn
from scipy import linalg
from stable_diffusion.constants.const import theme_available, class_available
import tqdm


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
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


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


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
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
    """Transfers elements to cuda if GPU is available."""
    if torch.cuda.is_available():
        return elements.to("cuda")
    return elements


class PartialInceptionNetwork(nn.Module):
    """A modified InceptionV3 network used for feature extraction."""
    def __init__(self):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)

    def output_hook(self, module, input, output):
        self.mixed_7c_output = output

    def forward(self, x):
        x = x * 2 - 1  # Normalize to [-1, 1]
        self.inception_network(x)
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def preprocess_image(im):
    """Preprocesses a single image."""
    assert im.shape[2] == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im).float()
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    return im


def preprocess_images(images, use_multiprocessing=False):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for im in images:
                job = pool.apply_async(preprocess_image, (im,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                im = job.get()
                final_images[idx] = im  # job.get()
    else:
        final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images



def get_activations(images, batch_size):
    """Calculates activations for last pool layer for all images."""
    num_images = images.shape[0]
    inception_network = PartialInceptionNetwork()
    inception_network = to_cuda(inception_network)
    inception_network.eval()
    n_batches = int(np.ceil(num_images / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)

    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)
        ims = images[start_idx:end_idx].to("cuda")
        with torch.no_grad():
            activations = inception_network(ims)
        inception_activations[start_idx:end_idx, :] = activations.cpu().numpy()

    return inception_activations


def calculate_activation_statistics(images, batch_size):
    """Calculates mean and covariance for FID."""
    act = get_activations(images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates Frechet Distance between two distributions."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ 
                               (sigma2 + eps * np.eye(sigma2.shape[0])))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid_value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid_value


def calculate_fid(images1, images2, use_multiprocessing=False, batch_size=64):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    images1 = preprocess_images(images1, use_multiprocessing)
    images2 = preprocess_images(images2, use_multiprocessing)
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    print("mu1", mu1.shape, "sigma1", sigma1.shape)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    print("mu2", mu2.shape, "sigma2", sigma2.shape)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

def load_style_generated_images(path, exclude="Abstractionism", seed=[188, 288, 588, 688, 888]):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """
    image_paths = []

    if exclude is not None:
        if exclude in theme_available:
            theme_tested = [x for x in theme_available]
            theme_tested.remove(exclude)
            class_tested = class_available
        else: # exclude is a class
            theme_tested = theme_available
            class_tested = [x for x in class_available]
            class_tested.remove(exclude)
    else:
        theme_tested = theme_available
        class_tested = class_available
    for theme in theme_tested:
        for object_class in class_tested:
            for individual in seed:
                image_paths.append(os.path.join(path, f"{theme}_{object_class}_seed{individual}.jpg"))
    if not os.path.isfile(image_paths[0]):
        raise FileNotFoundError(f"Could not find {image_paths[0]}")

    first_image = cv2.imread(image_paths[0])
    W, H = 512, 512
    image_paths.sort()
    image_paths = image_paths
    final_images = np.zeros((len(image_paths), H, W, 3), dtype=first_image.dtype)
    for idx, impath in tqdm(enumerate(image_paths)):
        im = cv2.imread(impath)
        im = cv2.resize(im, (W, H))  # Resize image to 512x512
        im = im[:, :, ::-1]  # Convert from BGR to RGB
        assert im.dtype == final_images.dtype
        final_images[idx] = im
    return final_images


def load_style_ref_images(path, exclude="Seed_Images"):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """
    image_paths = []

    if exclude is not None:
        # assert exclude in theme_available, f"{exclude} not in {theme_available}"
        if exclude in theme_available:
            theme_tested = [x for x in theme_available]
            theme_tested.remove(exclude)
            class_tested = class_available
        else: # exclude is a class
            theme_tested = theme_available
            class_tested = [x for x in class_available]
            class_tested.remove(exclude)
    else:
        theme_tested = theme_available
        class_tested = class_available

    for theme in theme_tested:
        for object_class in class_tested:
            for idx in range(1, 6):
                image_paths.append(os.path.join(path, theme, object_class, str(idx) + ".jpg"))

    first_image = cv2.imread(image_paths[0])
    W, H = 512, 512
    image_paths.sort()
    image_paths = image_paths
    final_images = np.zeros((len(image_paths), H, W, 3), dtype=first_image.dtype)
    for idx, impath in tqdm(enumerate(image_paths)):
        im = cv2.imread(impath)
        im = cv2.resize(im, (W, H))  # Resize image to 512x512
        im = im[:, :, ::-1]  # Convert from BGR to RGB
        assert im.dtype == final_images.dtype
        final_images[idx] = im
    return final_images