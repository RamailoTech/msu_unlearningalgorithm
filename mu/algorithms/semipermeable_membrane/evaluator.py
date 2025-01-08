import os
import logging
import torch
import numpy as np
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything
from mu.core.base_sampler import BaseSampler        
from stable_diffusion.constants.const import theme_available, class_available
from mu.helpers import load_config
from mu.helpers.utils import load_ckpt_from_config,load_style_generated_images,load_style_ref_images,calculate_fid
import timm
from tqdm import tqdm
from typing import Any, Dict
from torchvision import transforms
from torch.nn import functional as F
from mu.core.base_evaluator import BaseEvaluator
from transformers import CLIPTextModel, CLIPTokenizer  
from diffusers import (  
    UNet2DConditionModel,
    StableDiffusionPipeline,
    AltDiffusionPipeline,  
    DiffusionPipeline,  
) 
from mu.algorithms.semipermeable_membrane.src.models.spm import SPMNetwork, SPMLayer 
from mu.algorithms.semipermeable_membrane.src.models.model_util import load_checkpoint_model
from mu.algorithms.semipermeable_membrane.src.engine.train_util import encode_prompts
import safetensors
from pytorch_lightning import seed_everything
from safetensors.torch import load_file, save_file
from typing import Literal, Optional, List  
from mu.algorithms.semipermeable_membrane.src.models.merge_spm import load_state_dict, load_metadata_from_safetensors, merge_lora_models


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
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
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

    def _parse_precision(precision: str) -> torch.dtype:
        if precision == "fp32" or precision == "float32":
            return torch.float32
        elif precision == "fp16" or precision == "float16":
            return torch.float16
        elif precision == "bf16" or precision == "bfloat16":
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
        spm_paths = self.config["spm_paths"]
        v2 = self.config['v2']
        seed_everything(self.seed)

        spm_model_paths = [lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in spm_paths]
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
        output_dir = self.config["sampler_output_dir"]

        #make config directory
        config = f"{self.config["model_config"]}/{self.config["theme"]}/config.yaml"

        os.makedirs(output_dir,theme, exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")

        seed_everything(seed)
       
        with torch.no_grad():
            for test_theme in theme_available:
                for object_class in class_available:
                    prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."

                    prompt += config.unconditional_prompt
                    self.logger.info(f"Generating for prompt: {prompt}")
                    prompt_embeds, prompt_tokens = encode_prompts(
                        self.tokenizer, self.text_encoder, [prompt], return_tokens=True
                        )
                    if assigned_multipliers is not None:
                        multipliers = torch.tensor(assigned_multipliers).to("cpu", dtype=weight_dtype)
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
                    self.network.load_state_dict(weighted_spm)
                    with self.network:
                        image = self.pipe(
                            negative_prompt=config.negative_prompt,
                            width=config.widcth,
                            height=config.height,
                            num_inference_steps=config.num_inference_steps,
                            guidance_scale=config.guidance_scale,
                            generator=torch.cuda.manual_seed(seed),
                            num_images_per_prompt=config.generate_num,
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



class SemipermeableMembraneEvaluator(BaseEvaluator):
    """
    Example evaluator that calculates classification accuracy on generated images.
    Inherits from the abstract BaseEvaluator.
    """

    def __init__(self,config: Dict[str, Any], **kwargs):
        """
        Args:
            sampler (Any): An instance of a BaseSampler-derived class (e.g., SemipermeableMembraneSampler).
            config (Dict[str, Any]): A dict of hyperparameters / evaluation settings.
            **kwargs: Additional overrides for config.
        """
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampler = SemipermeableMembraneSampler(config)
        self.classification_model = None
        self.results = {}

    def load_model(self, *args, **kwargs):
        """
        Load the classification model for evaluation, using 'timm' 
        or any approach you prefer. 
        We assume your config has 'classification_ckpt' and 'task' keys, etc.
        """
        self.logger.info("Loading classification model...")
        model = self.config.get("classification_model")
        self.classification_model = timm.create_model(
            model, 
            pretrained=True
        ).to(self.device)
        task = self.config['task'] # "style" or "class"
        num_classes = len(theme_available) if task == "style" else len(class_available)
        self.classification_model.head = torch.nn.Linear(1024, num_classes).to(self.device)

        # Load checkpoint
        ckpt_path = self.config["classification_ckpt"]
        self.logger.info(f"Loading classification checkpoint from: {ckpt_path}")
        #NOTE: changed model_state_dict to state_dict as it was not present and added strict=False
        self.classification_model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["state_dict"],strict=False)
        self.classification_model.eval()
    
        self.logger.info("Classification model loaded successfully.")

    def preprocess_image(self, image: Image.Image):
        """
        Preprocess the input PIL image before feeding into the classifier.
        Replicates the transforms from your accuracy.py script.
        """
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return image_transform(image).unsqueeze(0).to(self.device)

    def calculate_accuracy(self, *args, **kwargs):
        """
        Calculate accuracy of the classification model on generated images.
        Mirrors the logic from your accuracy.py but integrated into a single method.
        """
        self.logger.info("Starting accuracy calculation...")

        # Pull relevant config
        theme = self.config.get("theme", None)
        input_dir = self.config['sampler_output_dir']
        output_dir = self.config["eval_output_dir"]
        seed_list = self.config.get("seed_list", [188, 288, 588, 688, 888])
        dry_run = self.config.get("dry_run", False)
        task = self.config['task']  

        if theme is not None:
            input_dir = os.path.join(input_dir, theme)
        else:
            input_dir = os.path.join(input_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = (os.path.join(output_dir, f"{theme}.pth") 
                       if theme is not None 
                       else os.path.join(output_dir, "result.pth"))

        # Initialize results dictionary
        self.results = {
            "test_theme": theme if theme is not None else "sd",
            "input_dir": input_dir,
        }

        if task == "style":
            self.results["loss"] = {th: 0.0 for th in theme_available}
            self.results["acc"] = {th: 0.0 for th in theme_available}
            self.results["pred_loss"] = {th: 0.0 for th in theme_available}
            self.results["misclassified"] = {
                th: {oth: 0 for oth in theme_available} 
                for th in theme_available
            }
        else:  # task == "class"
            self.results["loss"] = {cls_: 0.0 for cls_ in class_available}
            self.results["acc"] = {cls_: 0.0 for cls_ in class_available}
            self.results["pred_loss"] = {cls_: 0.0 for cls_ in class_available}
            self.results["misclassified"] = {
                cls_: {other_cls: 0 for other_cls in class_available} 
                for cls_ in class_available
            }

        # Evaluate
        if task == "style":
            for idx, test_theme in tqdm(enumerate(theme_available), total=len(theme_available)):
                theme_label = idx
                for seed in seed_list:
                    for object_class in class_available:
                        img_file = f"{test_theme}_{object_class}_seed{seed}.jpg"
                        img_path = os.path.join(input_dir, img_file)
                        if not os.path.exists(img_path):
                            self.logger.warning(f"Image not found: {img_path}")
                            continue

                        # Preprocess
                        image = Image.open(img_path)
                        tensor_img = self.preprocess_image(image)
                        label = torch.tensor([theme_label]).to(self.device)

                        # Forward pass
                        with torch.no_grad():
                            res = self.classification_model(tensor_img)

                        # Compute losses
                        loss = F.cross_entropy(res, label)
                        res_softmax = F.softmax(res, dim=1)
                        pred_loss_val = res_softmax[0][theme_label].item()
                        pred_label = torch.argmax(res).item()
                        pred_success = (pred_label == theme_label)

                        # Accumulate stats
                        self.results["loss"][test_theme] += loss.item()
                        self.results["pred_loss"][test_theme] += pred_loss_val
                        # Probability of success is 1 if pred_success else 0,
                        # but for your code, you were dividing by total. So let's keep a sum for now:
                        self.results["acc"][test_theme] += (1 if pred_success else 0)

                        misclassified_as = theme_available[pred_label]
                        self.results["misclassified"][test_theme][misclassified_as] += 1

                if not dry_run:
                    self.save_results(self.results, output_path)

        else: # task == "class"
            for test_theme in tqdm(theme_available, total=len(theme_available)):
                for seed in seed_list:
                    for idx, object_class in enumerate(class_available):
                        label_val = idx
                        img_file = f"{test_theme}_{object_class}_seed_{seed}.jpg"
                        img_path = os.path.join(input_dir, img_file)
                        if not os.path.exists(img_path):
                            self.logger.warning(f"Image not found: {img_path}")
                            continue

                        # Preprocess
                        image = Image.open(img_path)
                        tensor_img = self.preprocess_image(image)
                        label = torch.tensor([label_val]).to(self.device)

                        with torch.no_grad():
                            res = self.classification_model(tensor_img)

                        loss = F.cross_entropy(res, label)
                        res_softmax = F.softmax(res, dim=1)
                        pred_loss_val = res_softmax[0][label_val].item()
                        pred_label = torch.argmax(res).item()
                        pred_success = (pred_label == label_val)

                        self.results["loss"][object_class] += loss.item()
                        self.results["pred_loss"][object_class] += pred_loss_val
                        self.results["acc"][object_class] += (1 if pred_success else 0)

                        misclassified_as = class_available[pred_label]
                        self.results["misclassified"][object_class][misclassified_as] += 1

                if not dry_run:
                    self.save_results(self.results, output_path)

        self.logger.info("Accuracy calculation completed.")

    def calculate_fid_score(self, *args, **kwargs):
        """
        Calculate the Fréchet Inception Distance (FID) score using the images 
        generated by EraseDiffSampler vs. some reference images. 
        """
        self.logger.info("Starting FID calculation...")

        generated_path = self.config["sampler_output_dir"]  
        reference_path = self.config["reference_dir"]       
        forget_theme = self.config.get("forget_theme", None) 
        use_multiprocessing = self.config.get("multiprocessing", False)
        batch_size = self.config.get("batch_size", 64)
        output_dir = self.config["fid_output_path"]
        os.makedirs(output_dir, exist_ok=True)

        images_generated = load_style_generated_images(
            path=generated_path, 
            exclude=forget_theme, 
            seed=self.config.get("seed_list", [188, 288, 588, 688, 888])
        )
        images_reference = load_style_ref_images(
            path=reference_path, 
            exclude=forget_theme
        )

        fid_value = calculate_fid(
            images1=images_reference, 
            images2=images_generated, 
            use_multiprocessing=use_multiprocessing, 
            batch_size=batch_size
        )
        self.logger.info(f"Calculated FID: {fid_value}")
        self.results["FID"] = fid_value
        fid_path = os.path.join(output_dir, "fid_value.pth")
        torch.save({"FID": fid_value}, fid_path)
        self.logger.info(f"FID results saved to: {fid_path}")

    def save_results(self, results: dict, output_path: str):
        """
        Save evaluation results to a file. You can also do JSON or CSV if desired.
        """
        torch.save(results, output_path)
        self.logger.info(f"Results saved to: {output_path}")

    def run(self, *args, **kwargs):
        """
        Run the complete evaluation process:
        1) Load the classification model
        2) Generate images (using sampler)
        3) Calculate accuracy
        4) Calculate FID
        5) Save final results
        """

        # Call the sample method to generate images
        self.sampler.load_model()  
        self.sampler.sample()    

        # Load the classification model
        self.load_model()

        # Proceed with accuracy and FID calculations
        self.calculate_accuracy()
        self.calculate_fid_score()

        # Save results
        self.save_results(self.results, os.path.join(self.config["eval_output_dir"], "final_results.pth"))

        self.logger.info("Evaluation run completed.")

