# erase_diff/sampler.py

from core.base_sampler import BaseSampler
from mu.stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from algorithms.erase_diff.utils import sample_model
from algorithms.erase_diff.model import EraseDiffModel


class EraseDiffSampler(BaseSampler):
    """
    Sampler for the EraseDiff algorithm using DDIM.
    """

    def __init__(self, model: EraseDiffModel, config: dict, device: str):
        """
        Initialize the EraseDiffSampler.

        Args:
            model (EraseDiffModel): Instance of EraseDiffModel.
            config (dict): Configuration dictionary.
            device (str): Device to perform sampling on.
        """
        self.model = model
        self.config = config
        self.device = device
        self.ddim_steps = self.config.get('ddim_steps', 50)
        self.ddim_eta = self.config.get('ddim_eta', 0)
        self.sampler = DDIMSampler(self.model.model)

    def sample(self, c, h, w, scale, start_code=None, num_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
        """
        Generate samples using the DDIM sampler.

        Args:
            c (Any): Conditioning tensors.
            h (int): Height of the image.
            w (int): Width of the image.
            scale (float): Unconditional guidance scale.
            start_code (torch.Tensor, optional): Starting latent code. Defaults to None.
            num_samples (int, optional): Number of samples to generate. Defaults to 1.
            t_start (int, optional): Starting timestep. Defaults to -1.
            log_every_t (int, optional): Logging interval. Defaults to None.
            till_T (int, optional): Timestep to stop sampling. Defaults to None.
            verbose (bool, optional): Verbosity flag. Defaults to True.

        Returns:
            torch.Tensor: Generated samples.
        """
        samples = sample_model(
            self.model.model,
            self.sampler,
            c,
            h,
            w,
            self.ddim_steps,
            scale,
            self.ddim_eta,
            start_code=start_code,
            num_samples=num_samples,
            t_start=t_start,
            log_every_t=log_every_t,
            till_T=till_T,
            verbose=verbose
        )
        return samples
