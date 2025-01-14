#mu/algorithms/semipermeable_membrane/sampler.py

import os
import logging
from typing import Literal
from PIL import Image
from pathlib import Path

from pytorch_lightning import seed_everything
import torch

from transformers import CLIPTextModel, CLIPTokenizer  

from stable_diffusion.constants.const import theme_available, class_available
from mu.core.base_sampler import BaseSampler  
from mu.algorithms.semipermeable_membrane.src.configs.generation_config import GenerationConfig, load_config_from_yaml
from mu.algorithms.semipermeable_membrane.src.models.spm import SPMNetwork, SPMLayer 
from mu.algorithms.semipermeable_membrane.src.models.model_util import load_checkpoint_model
from mu.algorithms.semipermeable_membrane.src.engine.train_util import encode_prompts
from mu.algorithms.semipermeable_membrane.src.models.merge_spm import load_state_dict


#TODO remove this
theme_available = ['Abstractionism', 'Bricks', 'Cartoon']
class_available = ['Architectures', 'Bears', 'Birds']

MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]

class SemipermeableMembraneSampler(BaseSampler):
    """Semipermeable membrane Image Generator class extending a hypothetical BaseImageGenerator."""

    def __init__(self, config: dict, **kwargs):
        """
        Initialize the SemipermeableMembraneSampler with a YAML config (or dict).
        
        Args:
            config (Dict[str, Any]): Dictionary of hyperparams / settings.
            **kwargs: Additional keyword arguments that can override config entries.
        """
        super().__init__()

        self.config = config
        self.device = self.config['devices'][0]
        model_config = f"{self.config['model_config']}/{self.config['theme']}/config.yaml"
        self.model_config: GenerationConfig = load_config_from_yaml(model_config)
        self.model = None
        self.sampler = None
        self.network = None
        self.text_encoder = None
        self.tokenizer = None
        self.unet = None
        self.model_metadata = None
        self.erased_prompts_count = None
        self.pipe = None
        self.weight_dtype = None
        self.special_token_ids = None
        self.spms = None
        self.erased_prompt_embeds = None
        self.erased_prompt_tokens = None
        self.logger = logging.getLogger(__name__)

    def _parse_precision(self, precision: str) -> torch.dtype:
        if precision in ["fp32", "float32"]:
            return torch.float32
        elif precision in ["fp16", "float16"]:
            return torch.float16
        elif precision in ["bf16", "bfloat16"]:
            return torch.bfloat16
        raise ValueError(f"Invalid precision type: {precision}")
    
    def _text_tokenize(
            tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
            prompts: list[str],
        ):
        return tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
    
    def _text_encode(text_encoder: CLIPTextModel, tokens):
        return text_encoder(tokens.to(text_encoder.device))[0]
    
    def _encode_prompts(
            self,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTokenizer,
            prompts: list[str],
            return_tokens: bool = False,
        ):
        text_tokens = self._text_tokenize(tokenizer, prompts)
        text_embeddings = self._text_encode(text_encoder, text_tokens)

        if return_tokens:
            return text_embeddings, torch.unique(text_tokens, dim=1)
        return text_embeddings

    def _calculate_matching_score(
        prompt_tokens,
        prompt_embeds, 
        erased_prompt_tokens, 
        erased_prompt_embeds, 
        matching_metric: MATCHING_METRICS,
        special_token_ids: set[int],
        weight_dtype: torch.dtype = torch.float32,
    ):
        scores = []
        if "clipcos" in matching_metric:
            clipcos = torch.cosine_similarity(
                        prompt_embeds.flatten(1, 2), 
                        erased_prompt_embeds.flatten(1, 2), 
                        dim=-1).cpu()
            scores.append(clipcos)
        if "tokenuni" in matching_metric:
            prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
            tokenuni = []
            for ep in erased_prompt_tokens:
                ep_set = set(ep.tolist()) - special_token_ids
                tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
            scores.append(torch.tensor(tokenuni).to("cpu", dtype=weight_dtype))
        return torch.max(torch.stack(scores), dim=0)[0]

    def load_model(self) -> None:
        """
        Load the model using `config` and initialize the sampler.
        """
        self.logger.info("Loading model...")
        base_model = self.config["base_model"]
        spm_paths = self.config["spm_path"]
        v2 = self.config['v2']
        seed_everything(self.config['seed'])

        # spm_model_paths = [lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in spm_paths]
        spm_model_paths = [
            Path(lp) / f"{Path(lp).stem}_last.safetensors" if Path(lp).is_dir() else Path(lp)
            for lp in spm_paths
        ]
        self.weight_dtype = self._parse_precision(self.config["precision"])
        
        # load the pretrained SD
        tokenizer, text_encoder, unet, pipe = load_checkpoint_model(
            base_model,
            v2=v2,
            weight_dtype=self.weight_dtype
        )
        special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

        text_encoder.to(self.device, dtype=self.weight_dtype)
        text_encoder.eval()

        unet.to(self.device, dtype=self.weight_dtype)
        unet.enable_xformers_memory_efficient_attention()
        unet.requires_grad_(False)
        unet.eval()

        spms, metadatas = zip(*[
            load_state_dict(spm_model_path, self.weight_dtype) for spm_model_path in spm_model_paths
        ])
        # check if SPMs are compatible
        assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

        # get the erased concept
        erased_prompts = [md["prompts"].split(",") for md in metadatas]
        erased_prompts_count = [len(ep) for ep in erased_prompts]
        self.logger.info(f"Erased prompts: {erased_prompts}")

        erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
        erased_prompt_embeds, erased_prompt_tokens = encode_prompts(
            tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True
            )

        network = SPMNetwork(
            unet,
            rank=int(float(metadatas[0]["rank"])),
            alpha=float(metadatas[0]["alpha"]),
            module=SPMLayer,
        ).to(self.device, dtype=self.weight_dtype)

        self.network = network
        self.text_encoder = text_encoder
        self.erased_prompts_count = erased_prompts_count
        self.unet = unet
        self.tokenizer = tokenizer
        self.pipe = pipe
        self.special_token_ids = special_token_ids
        self.model_metadata = metadatas[0]
        self.spms = spms
        self.erased_prompt_embeds = erased_prompt_embeds
        self.erased_prompt_tokens = erased_prompt_tokens

        self.logger.info("Model loaded and sampler initialized successfully.")

    def sample(self) -> None:
        """
        Sample (generate) images using the loaded model and sampler, based on the config.
        """
        assigned_multipliers = self.config["spm_multiplier"]
        theme = self.config["theme"]           
        seed = self.config["seed"]
        output_dir = f"{self.config['sampler_output_dir']}/{theme}"

        #make config directory
        config = f"{self.config['model_config']}/{self.config['theme']}/config.yaml"

        os.makedirs(output_dir,exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")

        seed_everything(seed)
       
        with torch.no_grad():
            for test_theme in theme_available:
                for object_class in class_available:
                    prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."

                    prompt += self.model_config.unconditional_prompt
                    self.logger.info(f"Generating for prompt: {prompt}")
                    prompt_embeds, prompt_tokens = encode_prompts(
                        self.tokenizer, self.text_encoder, [prompt], return_tokens=True
                        )
                    if assigned_multipliers is not None:
                        multipliers = torch.tensor(assigned_multipliers).to("cpu", dtype=self.weight_dtype)
                        if assigned_multipliers == [0,0,0]:
                            matching_metric = "aazeros"
                        elif assigned_multipliers == [1,1,1]:
                            matching_metric = "zzone"
                    else:
                        multipliers = self._calculate_matching_score(
                            prompt_tokens,
                            prompt_embeds, 
                            self.erased_prompt_tokens, 
                            self.erased_prompt_embeds, 
                            matching_metric=matching_metric,
                            special_token_ids=self.special_token_ids,
                            weight_dtype=self.weight_dtype
                            )
                        multipliers = torch.split(multipliers, self.erased_prompts_count)
                    self.logger.info(f"multipliers: {multipliers}")
                    weighted_spm = dict.fromkeys(self.spms[0].keys())
                    used_multipliers = []
                    for spm, multiplier in zip(self.spms, multipliers):
                        max_multiplier = torch.max(multiplier)
                        for key, value in spm.items():
                            if weighted_spm[key] is None:
                                weighted_spm[key] = value * max_multiplier
                            else:
                                weighted_spm[key] += value * max_multiplier
                        used_multipliers.append(max_multiplier.item())
                    self.network.load_state_dict(weighted_spm,strict=False)
                    with self.network:
                        image = self.pipe(
                            negative_prompt=self.model_config.negative_prompt,
                            width=self.model_config.width,
                            height=self.model_config.height,
                            num_inference_steps=self.model_config.num_inference_steps,
                            guidance_scale=self.model_config.guidance_scale,
                            generator=torch.cuda.manual_seed(seed),
                            num_images_per_prompt=self.model_config.generate_num,
                            prompt_embeds=prompt_embeds,
                        ).images[0]
                        
                    self.save_image(image, os.path.join(output_dir, f"{test_theme}_{object_class}_seed_{seed}.jpg"))


        self.logger.info("Image generation completed.")

    def save_image(self, image: Image.Image, file_path: str) -> None:
        """
        Save an image to the specified path.
        """
        image.save(file_path)
        self.logger.info(f"Image saved at: {file_path}")


