from torchvision import transforms
from torchvision.transforms import InterpolationMode

class CenterSquareCrop:
    """
    Center crop the image to the smallest dimension.
    """

    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        return transforms.functional.crop(img, top=int(top), left=int(left), height=min_dim, width=min_dim)

def get_transform(interpolation: str = 'bicubic', size: int = 512) -> transforms.Compose:
    """
    Get a composed transformation pipeline.

    Args:
        interpolation (str, optional): Interpolation method. Defaults to 'bicubic'.
        size (int, optional): Resize size. Defaults to 512.

    Returns:
        transforms.Compose: Composed transformations.
    """
    interpolation_mode = {
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'lanczos': InterpolationMode.LANCZOS,
    }.get(interpolation.lower(), InterpolationMode.BICUBIC)

    return transforms.Compose([
        CenterSquareCrop(),
        transforms.Resize(size, interpolation=interpolation_mode),
    ])
