from mu.core import BaseSampler
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from mu.algorithms.esd.algorithm import ESDModel
from mu.helpers import sample_model

class ESDSampler(BaseSampler):
    """Sampler for the ESD algorithm."""

    def __init__(self, model: ESDModel, config: dict, device):
        self.model, self.model_orig = model.models
        self.config = config
        self.device = device
        self.ddim_steps = self.config['ddim_steps']
        self.ddim_eta = 0
        self.samplers = self.load_samplers(self.model, self.model_orig)

    def load_samplers(self, model, model_orig):
        """
        Load the samplers
        """
        sampler = DDIMSampler(model)
        sampler_orig = DDIMSampler(model_orig)
        return (sampler, sampler_orig)
            
    def sample(self, c, h, w, scale, start_code=None, num_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
        '''Generates samples using the model and sampler.

        Parameters:
            c (torch.Tensor): The conditioning input
            h (int): Height of the output image
            w (int): Width of the output image
            scale (float): The unconditional guidance scale
            start_code (torch.Tensor, optional): Starting noise tensor. Defaults to None.
            num_samples (int, optional): Number of samples to generate. Defaults to 1.
            t_start (int, optional): Starting timestep. Defaults to -1.
            log_every_t (int, optional): Log progress every t steps. Defaults to None.
            till_T (int, optional): Run sampling until timestep T. Defaults to None.
            verbose (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            torch.Tensor: Generated samples of shape (num_samples, channels, height, width)
        '''

        samples = sample_model(self.model, self.samplers[0], c, h, w, self.ddim_steps, scale, self.ddim_eta,
                               start_code=start_code, num_samples=num_samples, t_start=t_start,
                               log_every_t=log_every_t, till_T=till_T, verbose=verbose)
        return samples
    