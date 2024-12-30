# forget_me_not/datasets/forget_me_not_ti_dataset.py

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import mediapipe as mp


from mu.datasets import BaseDataset, UnlearnCanvasDataset, I2PDataset
from mu.helpers import read_text_lines
# Templates for object and style prompts
OBJECT_TEMPLATE = [
    "an image of {}"
]

STYLE_TEMPLATE = [
    "an image in {} Style",
]

I2P_TEMPLATE = [
    "an image of {}"
]

class ForgetMeNotTIDataset(BaseDataset):
    """
    Dataset for the Forget Me Not algorithm.
    Pre-processes images and tokenizes prompts for fine-tuning the model.
    """

    def __init__(
        self,
        processed_dataset_dir,
        dataset_type,
        template_name,
        template, 
        sample,
        tokenizer,
        token_map: Optional[dict] = None,
        class_data_root: Optional[str] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        h_flip: bool = True,
        color_jitter: bool = False,
        resize: bool = True,
        use_face_segmentation_condition: bool = False,
        blur_amount: int = 70
    ):
        """
        Initialize the ForgetMeNotDataset.

        Args:
            processed_dataset_dir (str): Path to the directory containing instance images.
            tokenizer: Tokenizer object for text prompts.
            token_map (Optional[dict]): A mapping from placeholder tokens to actual strings.
            class_data_root (Optional[str]): Path to the directory containing class images, if any.
            class_prompt (Optional[str]): Prompt for class images, if provided.
            size (int): Resolution to resize images to.
            h_flip (bool): Whether to apply horizontal flip augmentation.
            color_jitter (bool): Whether to apply color jitter augmentation.
            resize (bool): Whether to resize images.
            use_face_segmentation_condition (bool): Whether to use face segmentation conditioning.
            blur_amount (int): Amount of Gaussian blur to apply for face segmentation masks.
        """
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize
        self.h_flip = h_flip
        self.use_face_segmentation_condition = use_face_segmentation_condition
        self.blur_amount = blur_amount
        self.token_map = token_map

        # Validate and load instance images
        self.instance_data_root = Path(processed_dataset_dir) / template_name / "images.txt"
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")
        # self.instance_images_path = list(self.instance_data_root.iterdir())
        self.instance_images_path = read_text_lines(self.instance_data_root)
        self.num_instance_images = len(self.instance_images_path)

        if dataset_type == "unlearncanvas":
            if template == 'object': 
                self.templates = OBJECT_TEMPLATE
            elif template == 'style':
                self.templates = STYLE_TEMPLATE
        if dataset_type == "i2p":
            self.templates = I2P_TEMPLATE

        self._length = self.num_instance_images

        # If class data is provided, load and set up for prior preservation
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        # Define image transformations
        transform_list = []
        if resize:
            transform_list.append(
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
            )
        if color_jitter:
            transform_list.append(transforms.ColorJitter(0.1, 0.1))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.image_transforms = transforms.Compose(transform_list)

        # If using face segmentation, load mediapipe
        if self.use_face_segmentation_condition:
            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # Load instance image
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        # Prepare text prompt
        if self.templates and self.token_map is not None:
            input_tok = list(self.token_map.values())[0]
            text = random.choice(self.templates).format(input_tok)
        else:
            text = instance_image_path.stem
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text = text.replace(token, value)

        # Face segmentation conditioning if enabled
        if self.use_face_segmentation_condition:
            image = cv2.imread(str(instance_image_path))
            results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            black_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if results.detections:
                for detection in results.detections:
                    x_min = int(detection.location_data.relative_bounding_box.xmin * image.shape[1])
                    y_min = int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
                    width = int(detection.location_data.relative_bounding_box.width * image.shape[1])
                    height = int(detection.location_data.relative_bounding_box.height * image.shape[0])
                    black_image[y_min: y_min + height, x_min: x_min + width] = 255

            # Apply Gaussian blur to the mask
            black_mask = Image.fromarray(black_image, mode="L").filter(
                ImageFilter.GaussianBlur(radius=self.blur_amount)
            )
            black_mask = transforms.ToTensor()(black_mask)
            black_mask = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)(black_mask)
            example["mask"] = black_mask

        # Apply random horizontal flip
        if self.h_flip and random.random() > 0.5:
            hflip = transforms.RandomHorizontalFlip(p=1)
            example["instance_images"] = hflip(example["instance_images"])
            if self.use_face_segmentation_condition and "mask" in example:
                example["mask"] = hflip(example["mask"])

        # Tokenize prompts
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # Unconditional prompt
        example["uncond_prompt_ids"] = self.tokenizer(
            "",
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # If class images are used, load and tokenize class prompts
        if self.class_data_root:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image_path).convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example
