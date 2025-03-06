from T2IBenchmark import calculate_fid

def fid_score(generated_image_dir, reference_image_dir, device='cuda', seed=42, batch_size=128, dataloader_workers=16, verbose=True):
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
