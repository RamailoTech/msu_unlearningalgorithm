# forget_me_not/datasets/forget_me_not_dataset.py

from pathlib import Path

from torchvision import transforms
import torch 

from lora_diffusion.patch_lora import safe_open, parse_safeloras_embeds, apply_learned_embed_in_clip

from mu.algorithms.forget_me_not.model import ForgetMeNotModel
from mu.datasets import BaseDataset
from PIL import Image
from mu.helpers import read_text_lines



class ForgetMeNotDataset(BaseDataset):
    """
    Dataset for the Forget Me Not algorithm.
    Pre-processes images and tokenizes prompts for fine-tuning the model.
    """

    def __init__(
        self,
        config,
        processed_dataset_dir,
        dataset_type,
        template_name,
        template, 
        use_sample,
        tokenizer,
        size=512,
        center_crop=False,
        use_added_token=True,
        use_pooler=False,
        use_ti: bool = True,
        model: ForgetMeNotModel = None,
    ):
        """
        Initialize the ForgetMeNotDataset.

        Args:
            processed_dataset_dir (str): Path to the directory containing instance images.
            tokenizer: Tokenizer object for text prompts.
            token_map (Optional[dict]): A mapping from placeholder tokens to actual strings.
        """
        self.use_added_token = use_added_token
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path = []
        self.instance_prompt = []
        self.instance_data_root = Path(processed_dataset_dir) / template_name / "images.txt"
        self.instance_images_path = read_text_lines(self.instance_data_root)

        # print(f"***************************************", multi_concept, "***************************************")

        concept = None

        tok_idx = 1
        token = None
        idempotent_token = True
        safeloras = safe_open(config.get('ti_weights_path'), framework="pt", device="cpu")
        tok_dict = parse_safeloras_embeds(safeloras)

        tok_dict = {f"<s{tok_idx + i}>": tok_dict[k] for i, k in enumerate(sorted(tok_dict.keys()))}
        tok_idx += len(tok_dict.keys())
        if dataset_type == "unlearncanvas":
            if use_ti:
                concept = [template_name, template , len(tok_dict.keys())]
            else: 
                concept = [template_name, template, -1]
        
        if dataset_type == "i2p":
            if use_ti:
                concept = [template_name, template , len(tok_dict.keys())]
            else: 
                concept = [template_name, template , -1 ]

        # print("---Adding Tokens---:", c, t)
        apply_learned_embed_in_clip(
            tok_dict,
            model.text_encoder,
            tokenizer,
            token=token,
            idempotent=idempotent_token,
        )
        c, t, num_tok = concept

        p = Path(processed_dataset_dir, c) / "images.txt"
        if not p.exists():
            raise ValueError(f"Instance {p} images root doesn't exists.")

        # image_paths = list(p.iterdir()) 
        # print(f"***************************************", image_paths, "***************************************")
        # self.instance_images_path = image_paths
     
        # prompt_lines = read_text_lines(Path(processed_dataset_dir, c, "prompts.txt"))

        # # Generate target tokens and associate with prompts
        # for i, prompt in enumerate(prompt_lines):
        #     # Generate a target snippet based on the current index or other logic
        #     target_snippet = f"{''.join([f'<s{i + 1}>' for i in range(num_tok)])}" if use_added_token else c.replace("-", " ")
            
        #     # Append the prompt and target tokens as a tuple
        #     self.instance_prompt.append((prompt, target_snippet))
            
        target_snippet = f"{''.join([f'<s{tok_idx + i}>' for i in range(num_tok)])}" if use_added_token else c.replace(
            "-", " ")
        if t == "object":
            self.instance_prompt += [(f"a {target_snippet} image", target_snippet)] * len(self.instance_images_path)
        elif t == "style":
            self.instance_prompt += [(f"an image in {target_snippet} Style", target_snippet)] * len(
                self.instance_images_path)
            
        elif t == "i2p":
            self.instance_prompt += [(f"a {target_snippet} image", target_snippet)] * len(
                self.instance_images_path)
        else:
            raise ValueError("unknown concept type!")
        if use_added_token:
            tok_idx += num_tok
        
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_prompt, target_tokens = self.instance_prompt[index % self.num_instance_images]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(
            target_tokens,
            add_special_tokens=False
        ).input_ids

        pooler_token_id = self.tokenizer(
            "<|endoftext|>",
            add_special_tokens=False
        ).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1] * len(concept_ids)
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]

        return example