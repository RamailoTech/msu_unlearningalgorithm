# mu_attack/core/sd_interfaces.py

from abc import ABC, abstractmethod
import logging
import torch
import torch.nn.functional as F



class BaseStableDiffusionPipeline(ABC):
    """Interface for a Stable Diffusion pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.object_list = ['cassette_player', 'church', 'english_springer', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench', "chain_saw"]
        self.object_labels = [482, 497, 217, 566, 569, 571, 574, 701, 0, 491]

    @abstractmethod
    def load_model(self):
        """Load model weights/config."""
        pass


    @abstractmethod
    def sample(
        self, *args, **kwargs
    ):
        """Run the diffusion sampling and return PIL or numpy image(s)."""
        pass
