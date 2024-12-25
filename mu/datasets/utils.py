import torchvision.transforms as torch_transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

class CenterSquareCrop:
    def __call__(self, img):
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        return F.crop(img, top=int(top), left=int(left), height=min_dim, width=min_dim)

def get_transform(interpolation=InterpolationMode.BICUBIC, size: int = 512) -> torch_transforms.Compose:
    """
    Get a composed transformation pipeline.

    Args:
        interpolation (str, optional): Interpolation method. Defaults to 'bicubic'.
        size (int, optional): Resize size. Defaults to 512.

    Returns:
        transforms.Compose: Composed transformations.
    """
    transform = torch_transforms.Compose([
        CenterSquareCrop(),
        torch_transforms.Resize(size, interpolation=interpolation),
    ])
    return transform
