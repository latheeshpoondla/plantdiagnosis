"""torchvision transform pipelines for train/eval.

Import of torch/torchvision is deferred into the function body so that this
module can still be imported (and its constants used) on a machine that only
runs the data-scan stage and has no torch installed -- the actual transform
objects are only built once you're in an environment (Kaggle) that has
torchvision.
"""
from __future__ import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# backbone -> the image size it was pretrained at (ConvNeXt-Tiny and
# Swin-Tiny both use 224 in their standard torchvision/timm ImageNet weights)
BACKBONE_IMAGE_SIZE = {
    "convnext_tiny": 224,
    "swin_tiny": 224,
}


def build_transforms(
    split: str,
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    """split: 'train' or 'eval'. Returns a torchvision.transforms.Compose."""
    from torchvision import transforms as T

    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    if split == "eval":
        resize_to = int(round(image_size * 256 / 224))  # keep the standard 224/256 ratio
        return T.Compose([
            T.Resize(resize_to),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
