#evaluation/metrics/fid.py

import os

from T2IBenchmark import calculate_fid

def fid_score(generated_image_dir, reference_image_dir, device='cuda', seed=42, batch_size=128, dataloader_workers=16, verbose=True):
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
