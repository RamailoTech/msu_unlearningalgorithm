import torch
from omegaconf import OmegaConf
from stable_diffusion.ldm.util import instantiate_from_config
from helpers import load_model_from_config

class ESDModel(UnlearningModel):
    """
    ESD-specific implementation of the UnlearningModel.
    """

    def __init__(self, config_path, ckpt_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(config_path, ckpt_path, device)

    def load_model(self, config, ckpt_path, device="cpu", verbose=False):
        config = load_model_from_config(config)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        model = instantiate_from_config(config.model)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

    def get_learned_conditioning(self, prompts):
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z, conditioning):
        return self.model.apply_model(z, conditioning)
