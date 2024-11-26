from core.base_sampler import BaseSampler
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from algorithms.esd.utils import sample_model
from algorithms.esd.algorithm import ESDModel

class ESDSampler(BaseSampler):
    """Sampler for the ESD algorithm."""

    def __init__(self, model: ESDModel, config: dict, device):
        self.model = model
        self.config = config
        self.device = device
        self.ddim_steps = self.config['ddim_steps']
        self.ddim_eta = 0
        self.sampler = DDIMSampler(self.model.model)

    def sample(self, c, h, w, scale, start_code=None, num_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
        samples = sample_model(self.model.model, self.sampler, c, h, w, self.ddim_steps, scale, self.ddim_eta,
                               start_code=start_code, n_samples=num_samples, t_start=t_start,
                               log_every_t=log_every_t, till_T=till_T, verbose=verbose)
        return samples
    