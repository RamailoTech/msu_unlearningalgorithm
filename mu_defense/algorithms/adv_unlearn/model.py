import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from mu_attack.tasks.utils.text_encoder import CustomTextEncoder
from mu_defense.algorithms.adv_unlearn import get_models

from mu_defense.core import BaseModel 

class AdvUnlearnModel(BaseModel):
    """
    AdvUnlearnModel handles loading of the components for adversarial unlearning.
    This includes:
        - The VAE from the pretrained model.
        - The tokenizer.
        - The text encoder (and its custom wrapper).
        - The diffusion models (trainable and frozen versions) along with their samplers.
    """
    def __init__(
        self,
        model_name_or_path: str,
        config_path: str,
        compvis_ckpt_path: str,
        cache_path: str,
        devices: list
    ):
        """
        Initialize the AdvUnlearnModel loader.

        Args:
            model_name_or_path (str): Path or identifier of the pretrained model.
            config_path (str): Path to the model configuration file.
            compvis_ckpt_path (str): Path to the model checkpoint.
            cache_path (str): Directory for caching downloaded models.
            devices (list): List of device strings (e.g., ['cuda:0', 'cuda:1']) for model placement.
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.config_path = config_path
        self.ckpt_path = compvis_ckpt_path
        self.cache_path = cache_path
        self.devices = devices

        # Load the VAE.
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name_or_path,
            subfolder="vae",
            cache_dir=self.cache_path
        ).to(self.devices[0])

        # Load the tokenizer.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name_or_path,
            subfolder="tokenizer",
            cache_dir=self.cache_path
        )

        # Load the text encoder and wrap it with your custom encoder.
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name_or_path,
            subfolder="text_encoder",
            cache_dir=self.cache_path
        ).to(self.devices[0])
        self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.devices[0])
        self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

        # Load diffusion models using your helper function.
        self.model_orig, self.sampler_orig, self.model, self.sampler = get_models(
            self.config_path,
            self.compvis_ckpt_path,
            self.devices
        )
        self.model_orig.eval()  # Set the frozen model to evaluation mode.

    def save_model(self, model: torch.nn.Module, output_path: str) -> None:
        """
        Save the model's state dictionary.

        Args:
            model (torch.nn.Module): The model to be saved.
            output_path (str): The file path where the model checkpoint will be stored.
        """
        torch.save({"state_dict": model.state_dict()}, output_path)

    def get_learned_conditioning(self, prompts: list):
        """
        Obtain the learned conditioning for the given prompts using the trainable diffusion model.

        Args:
            prompts (list): A list of prompt strings.

        Returns:
            The conditioning tensors produced by the model.
        """
        return self.model.get_learned_conditioning(prompts)

    def apply_model(self, z: torch.Tensor, t: torch.Tensor, c):
        """
        Apply the diffusion model to produce an output.

        Args:
            z (torch.Tensor): Noisy latent vectors.
            t (torch.Tensor): Timestep tensor.
            c: Conditioning tensors.

        Returns:
            torch.Tensor: The output of the diffusion model.
        """
        return self.model.apply_model(z, t, c)
