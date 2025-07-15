#evaluation/metrics/fid.py

import os

from T2IBenchmark import calculate_fid

def fid_score(generated_image_dir, reference_image_dir="data/coco_reference_images/coco_10k", device='cuda', seed=42, batch_size=128, dataloader_workers=16, verbose=True):
    """_summary_

    Args:
        generated_image_dir (str): Path to the directory containing generated images.
        reference_image_dir (str, optional): Path to directory containing referenced images. Defaults to "data/coco_reference_images/coco_10k".
        device (str, optional):  Device to run the computation on (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        batch_size (int, optional): Batch size for processing images. Defaults to 128.
        dataloader_workers (int, optional): Number of workers for the dataloader. Defaults to 16.
        verbose (bool, optional):  If True, prints detailed information during computation. Defaults to True.

    Returns:
        _type_: _description_
    """
    valid_extensions = ('.png', '.jpg', '.jpeg')
    generated_images = [f for f in os.listdir(generated_image_dir) if f.lower().endswith(valid_extensions)]
    reference_images = [f for f in os.listdir(reference_image_dir) if f.lower().endswith(valid_extensions)]

    assert len(generated_images) == len(reference_images), (
        f"Mismatch in number of images: {len(generated_images)} in generated and {len(reference_images)} in reference directory."
    )

    fid_score = calculate_fid(
        generated_image_dir,
        reference_image_dir,
        device=device,
        seed=seed,
        batch_size=batch_size,
        dataloader_workers=dataloader_workers,
        verbose=verbose
    )
    return fid_score
