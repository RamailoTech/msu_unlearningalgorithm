from algorithms.core.base_sampler import BaseSampler
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler

class ESDSampler(BaseSampler):
    def __init__(self, model):
        self.model = model
        self.sampler = DDIMSampler(model.model)

    def sample(self, num_samples: int, conditioning, **kwargs):
        samples, intermediates = self.sampler.sample(
            conditioning=conditioning,
            batch_size=num_samples,
            **kwargs
        )
        return samples
