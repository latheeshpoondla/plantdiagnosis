"""Everything numeric about the resolved manifests: class counts per split,
image dimension sampling, an optional corrupt-file sweep, and an optional
sampled channel mean/std (in case ImageNet normalization stats turn out to
be a poor fit for this data -- worth checking once, not assuming)."""
from __future__ import annotations

import random

import pandas as pd
from PIL import Image, UnidentifiedImageError


def class_counts(df: pd.DataFrame, class_col: str = "joint_key") -> dict:
    out = {}
    for split, sdf in df.groupby("split"):
        out[split] = sdf[class_col].value_counts().sort_index().to_dict()
    return out


def image_size_sample(df: pd.DataFrame, n: int = 300, seed: int = 42) -> dict:
    sample = df.sample(min(n, len(df)), random_state=seed)
    widths, heights, failed = [], [], []
    for fp in sample["filepath"]:
        try:
            with Image.open(fp) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except (UnidentifiedImageError, OSError) as e:
            failed.append({"filepath": fp, "error": str(e)})
    if not widths:
        return {"n_sampled": len(sample), "n_readable": 0, "failed": failed}
    return {
        "n_sampled": len(sample),
        "n_readable": len(widths),
        "width_min": min(widths), "width_max": max(widths),
        "width_mean": round(sum(widths) / len(widths), 1),
        "height_min": min(heights), "height_max": max(heights),
        "height_mean": round(sum(heights) / len(heights), 1),
        "failed": failed,
    }


def corrupt_check(df: pd.DataFrame, sample_n: int | None = None, seed: int = 42) -> list[dict]:
    """Opens (and calls .verify() on) every image, or a random sample of
    `sample_n` if given. Returns the list of files that fail to open/verify."""
    target = df if sample_n is None else df.sample(min(sample_n, len(df)), random_state=seed)
    bad = []
    for fp in target["filepath"]:
        try:
            with Image.open(fp) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a health check
            bad.append({"filepath": fp, "error": str(e)})
    return bad


def channel_mean_std(df: pd.DataFrame, n: int = 500, seed: int = 42) -> dict:
    """Sampled per-channel mean/std in [0,1], for comparing against the
    ImageNet defaults this project uses by default (see transforms.py)."""
    import numpy as np

    sample = df.sample(min(n, len(df)), random_state=seed)
    sums = np.zeros(3)
    sums_sq = np.zeros(3)
    n_pixels = 0
    n_read = 0
    for fp in sample["filepath"]:
        try:
            with Image.open(fp) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        except (UnidentifiedImageError, OSError):
            continue
        n_read += 1
        sums += arr.sum(axis=(0, 1))
        sums_sq += (arr ** 2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]
    if n_pixels == 0:
        return {"n_sampled": len(sample), "n_readable": 0}
    mean = sums / n_pixels
    var = sums_sq / n_pixels - mean ** 2
    std = [float(x) for x in (var.clip(min=0)) ** 0.5]
    return {
        "n_sampled": len(sample),
        "n_readable": n_read,
        "mean_rgb": [round(float(x), 4) for x in mean],
        "std_rgb": [round(float(x), 4) for x in std],
        "imagenet_mean_rgb": [0.485, 0.456, 0.406],
        "imagenet_std_rgb": [0.229, 0.224, 0.225],
    }
