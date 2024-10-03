from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler

class ESDSampler(UnlearningSampler):
    """
    ESD-specific implementation of the UnlearningSampler.
    """

    def __init__(self, model):
        self.sampler = DDIMSampler(model)

    def sample(self, model, **kwargs):
        pass
