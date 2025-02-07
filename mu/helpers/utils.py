from typing import List, Any
import argparse
from omegaconf import OmegaConf
import torch
from pytorch_lightning.utilities.distributed import rank_zero_only
from pathlib import Path
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


def param_choices(model, train_method, component="all", final_layer_norm=False):
    # choose parameters to train based on train_method
    parameters = []

    # Text Encoder FUll Weight Tuning
    if train_method == "text_encoder_full":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Final Layer Norm
            if name.startswith("final_layer_norm"):
                if component == "all" or final_layer_norm == True:
                    print(name)
                    parameters.append(param)
                else:
                    pass

            # Transformer layers
            elif name.startswith("encoder"):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            # Embedding layers
            else:
                pass

    # Text Encoder Layer 0 Tuning
    elif train_method == "text_encoder_layer0":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith("encoder.layers.0"):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer01":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith("encoder.layers.0") or name.startswith(
                "encoder.layers.1"
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass
            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer012":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer0123":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer01234":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer012345":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer0123456":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
                or name.startswith("encoder.layers.6")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer01234567":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
                or name.startswith("encoder.layers.6")
                or name.startswith("encoder.layers.7")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer012345678":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
                or name.startswith("encoder.layers.6")
                or name.startswith("encoder.layers.7")
                or name.startswith("encoder.layers.8")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer0123456789":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
                or name.startswith("encoder.layers.6")
                or name.startswith("encoder.layers.7")
                or name.startswith("encoder.layers.8")
                or name.startswith("encoder.layers.9")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer012345678910":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
                or name.startswith("encoder.layers.6")
                or name.startswith("encoder.layers.7")
                or name.startswith("encoder.layers.8")
                or name.startswith("encoder.layers.9")
                or name.startswith("encoder.layers.10")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer01234567891011":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.3")
                or name.startswith("encoder.layers.4")
                or name.startswith("encoder.layers.5")
                or name.startswith("encoder.layers.6")
                or name.startswith("encoder.layers.7")
                or name.startswith("encoder.layers.8")
                or name.startswith("encoder.layers.9")
                or name.startswith("encoder.layers.10")
                or name.startswith("encoder.layers.11")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer0_11":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if name.startswith("encoder.layers.0") or name.startswith(
                "encoder.layers.11"
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer01_1011":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.10")
                or name.startswith("encoder.layers.11")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    elif train_method == "text_encoder_layer012_91011":
        for name, param in model.text_encoder.text_model.named_parameters():
            # Encoder Layer 0
            if (
                name.startswith("encoder.layers.0")
                or name.startswith("encoder.layers.1")
                or name.startswith("encoder.layers.2")
                or name.startswith("encoder.layers.9")
                or name.startswith("encoder.layers.10")
                or name.startswith("encoder.layers.11")
            ):
                if component == "ffn" and "mlp" in name:
                    print(name)
                    parameters.append(param)
                elif component == "attn" and "self_attn" in name:
                    print(name)
                    parameters.append(param)
                elif component == "all":
                    print(name)
                    parameters.append(param)
                else:
                    pass

            elif name.startswith("final_layer_norm") and final_layer_norm == True:
                print(name)
                parameters.append(param)

            else:
                pass

    # UNet Model Tuning
    else:
        for name, param in model.model.diffusion_model.named_parameters():
            # train all layers except x-attns and time_embed layers
            if train_method == "noxattn":
                if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                    pass
                else:
                    print(name)
                    parameters.append(param)

            # train only self attention layers
            if train_method == "selfattn":
                if "attn1" in name:
                    print(name)
                    parameters.append(param)

            # train only x attention layers
            if train_method == "xattn":
                if "attn2" in name:
                    print(name)
                    parameters.append(param)

            # train all layers
            if train_method == "full":
                print(name)
                parameters.append(param)

            # train all layers except time embed layers
            if train_method == "notime":
                if not (name.startswith("out.") or "time_embed" in name):
                    print(name)
                    parameters.append(param)
            if train_method == "xlayer":
                if "attn2" in name:
                    if "output_blocks.6." in name or "output_blocks.8." in name:
                        print(name)
                        parameters.append(param)
            if train_method == "selflayer":
                if "attn1" in name:
                    if "input_blocks.4." in name or "input_blocks.7." in name:
                        print(name)
                        parameters.append(param)

    return parameters
