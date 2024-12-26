import os
from pathlib import Path
from typing import Any, Callable, List, Tuple

import torch
from PIL import Image
from src import utils
from src.utils import safe_dir
from torch.utils.data import Dataset

# Import your filtering logic if available
# from src.filter import filter as filter_fn


class ConceptAblationDataset(Dataset):
    """
    Dataset for the Concept Ablation algorithm.
    Integrates prompt reading, image generation, and optional filtering.
    """

    def __init__(
        self,
        concept_type: str,
        prompts_path: str,
        output_dir: str,
        base_config: str,
        resume_ckpt: str,
        delta_ckpt: str = None,
        caption_target: str = None,
        train_size: int = 1000,
        n_samples: int = 10,
        image_size: int = 512,
        transform: Callable = None,
        use_regularization: bool = False,
    ):
        """
        Initialize the ConceptAblationDataset.

        Args:
            concept_type (str): Type of concept being removed (e.g., 'style', 'object', 'memorization').
            prompts_path (str): Path to a text file containing the initial prompts to generate images from.
            output_dir (str): Directory where generated/filtered images and files will be stored.
            base_config (str): Base config path for Stable Diffusion model.
            resume_ckpt (str): Checkpoint for the Stable Diffusion model.
            delta_ckpt (str, optional): Delta checkpoint for additional fine-tuning. Defaults to None.
            caption_target (str, optional): Target style or concept to remove, used in filtering. Defaults to None.
            train_size (int, optional): Number of images to generate. Defaults to 1000.
            n_samples (int, optional): Number of images per prompt batch generation step. Defaults to 10.
            image_size (int, optional): Image resolution. Defaults to 512.
            transform (Callable, optional): Transformations to apply to images. Defaults to None.
            use_regularization (bool, optional): If True, add a second "regularization" dataset. Defaults to False.
        """
        self.concept_type = concept_type
        self.prompts_path = prompts_path
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.resume_ckpt = resume_ckpt
        self.delta_ckpt = delta_ckpt
        self.caption_target = caption_target
        self.train_size = train_size
        self.n_samples = n_samples
        self.image_size = image_size
        self.transform = transform
        self.use_regularization = use_regularization

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and filter data if needed (creates images.txt and prompts.txt)
        self._generate_and_filter_data()

        # After generation and filtering, load final images and prompts
        self.image_paths, self.prompts = self._load_final_data()

    def _generate_and_filter_data(self):
        """
        Handle the logic for generating and filtering images from prompts.
        This method encapsulates what was previously done in preprocessing.py and train.py.
        It checks if final dataset files exist (images.txt, prompts.txt) and, if not, generates them.

        Steps:
        1. Read initial prompts from prompts_path.
        2. Depending on concept_type:
           - If 'memorization', follow specific logic (generate anchor/target, filter).
           - Otherwise, directly generate images for the prompts.
        3. Possibly apply filtering, save final prompts and images to images.txt, prompts.txt.
        """

        images_file = self.output_dir / "images.txt"
        prompts_file = self.output_dir / "prompts.txt"

        # If final data files already exist, no need to regenerate
        if images_file.exists() and prompts_file.exists():
            return

        # Load initial prompts
        with open(self.prompts_path, "r") as f:
            raw_prompts = f.read().splitlines()

        if self.concept_type == "memorization":
            # For memorization, we apply the logic from preprocessing:
            #  - Generate initial samples
            #  - Filter them, produce anchor and target prompts
            #  - Generate anchor dataset
            #  - Final filter to produce images.txt and prompts.txt

            # Example logic (pseudo-code, adapt as needed):
            # 1. Generate 5x each prompt for checking
            prompt_size = len(raw_prompts)
            data_for_check = [5 * [p] for p in raw_prompts]
            data_for_check = self._chunk_list(data_for_check, self.n_samples)

            check_dir = safe_dir(self.output_dir / "check")
            check_samples = safe_dir(check_dir / "samples")

            # If not enough samples generated, call distributed_sample_images
            if len(list(check_samples.glob("*"))) != prompt_size * 5:
                utils.distributed_sample_images(
                    data_for_check,
                    # ranks from GPU devices, adapt as needed:
                    ranks=[0],
                    base_config=self.base_config,
                    resume_ckpt=self.resume_ckpt,
                    delta_ckpt=self.delta_ckpt,
                    outdir=str(check_dir),
                    steps=100,
                )

            # Filtering step (depends on your filtering logic)
            # filter_dir = safe_dir(self.output_dir / 'filtered')
            # unfilter_dir = safe_dir(self.output_dir / 'unfiltered')
            # anchor_prompts, target_prompts = filter_fn(str(check_dir), str(filter_dir), str(unfilter_dir), self.caption_target)

            # For demonstration, let's assume anchor_prompts and target_prompts are derived somehow:
            anchor_prompts = raw_prompts  # Replace with real filtered results
            target_prompts = [self.caption_target] if self.caption_target else []

            # Generate anchor dataset
            assert self.train_size % len(anchor_prompts) == 0
            n_repeat = self.train_size // len(anchor_prompts)
            anchor_data = [n_repeat * [p] for p in anchor_prompts]
            anchor_data = self._chunk_list(anchor_data, self.n_samples)

            anchor_dir = safe_dir(self.output_dir / "anchor")
            anchor_samples = safe_dir(anchor_dir / "samples")
            if len(list(anchor_samples.glob("*"))) != self.train_size:
                utils.distributed_sample_images(
                    anchor_data,
                    ranks=[0],
                    base_config=self.base_config,
                    resume_ckpt=self.resume_ckpt,
                    delta_ckpt=self.delta_ckpt,
                    outdir=str(anchor_dir),
                    steps=200,
                )

            # Final filtering step to produce final dataset:
            # unfiltered_anchor_dir = safe_dir(self.output_dir / 'un_filtered_anchor')
            # filter_fn(str(anchor_dir), str(self.output_dir), str(unfiltered_anchor_dir), self.caption_target)

            # Let's assume after filtering, 'images.txt' and 'prompts.txt' are written
            # For demonstration, write them directly:
            with open(prompts_file, "w") as pf, open(images_file, "w") as imf:
                # In a real scenario, you'd write filtered prompts and corresponding image paths
                # Here we just simulate by writing the anchor dataset images:
                for i in range(self.train_size):
                    # Dummy image and prompt path (replace with actual results)
                    img_path = anchor_dir / "samples" / f"{i}.png"
                    prompt = anchor_prompts[i % len(anchor_prompts)]
                    imf.write(str(img_path) + "\n")
                    pf.write(prompt + "\n")

        else:
            # For 'style', 'object', or other concept_types:
            # Generate images directly for all prompts
            assert self.train_size % len(raw_prompts) == 0
            n_repeat = self.train_size // len(raw_prompts)
            data = [n_repeat * [p] for p in raw_prompts]
            data = self._chunk_list(data, self.n_samples)

            sample_path = safe_dir(self.output_dir / "samples")
            existing_count = len(list(sample_path.glob("*")))
            if existing_count != self.train_size:
                utils.distributed_sample_images(
                    data,
                    ranks=[0],
                    base_config=self.base_config,
                    resume_ckpt=self.resume_ckpt,
                    delta_ckpt=self.delta_ckpt,
                    outdir=str(self.output_dir),
                    steps=200,
                )

            # Write images.txt and prompts.txt
            # Assume distributed_sample_images produced images in `outdir/samples`
            generated_images = sorted((self.output_dir / "samples").glob("*"))
            if len(generated_images) != self.train_size:
                raise RuntimeError("Mismatch in the number of generated images.")
            with open(images_file, "w") as imf, open(prompts_file, "w") as pf:
                for i, img_path in enumerate(generated_images):
                    imf.write(str(img_path) + "\n")
                    pf.write(raw_prompts[i % len(raw_prompts)] + "\n")

    def _load_final_data(self) -> Tuple[List[str], List[str]]:
        """
        Load the final images.txt and prompts.txt after generation and filtering.
        """
        images_path = self.output_dir / "images.txt"
        prompts_path = self.output_dir / "prompts.txt"
        if not (images_path.exists() and prompts_path.exists()):
            raise FileNotFoundError(
                "Images or prompts file not found after generation and filtering step."
            )

        with open(images_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]

        with open(prompts_path, "r") as f:
            prompts = [line.strip() for line in f.readlines()]

        if len(image_paths) != len(prompts):
            raise ValueError(
                "Mismatch between number of images and prompts in the final dataset."
            )

        return image_paths, prompts

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        img_path = self.image_paths[idx]
        prompt = self.prompts[idx]

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, prompt

    @staticmethod
    def _chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
        """
        Chunk a list into sub-lists of a given size.
        """
        # Flatten the nested list first if needed
        flattened = [item for sub in lst for item in sub]
        return [
            flattened[i : i + chunk_size] for i in range(0, len(flattened), chunk_size)
        ]
