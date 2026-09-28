"""Walks the resolved dataset roots and builds a flat manifest (one row per
image) with the raw class name resolved into crop / disease / joint labels
via the taxonomy. This is the only place that touches the filesystem layout
of the raw datasets -- everything downstream works off the manifest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.taxonomy import Taxonomy
from src.utils.io_utils import list_images


def scan_plantvillage(
    variant_root: Path,
    taxonomy: Taxonomy,
    extensions: list[str],
    variant_name: str = "color",
) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in variant_root.iterdir() if p.is_dir()):
        raw_class = class_dir.name
        try:
            meta = taxonomy.lookup("plantvillage", raw_class)
        except KeyError:
            # A folder Kaggle shipped that isn't one of our known 38 classes
            # (e.g. a stray README dir). Skip it loudly rather than crash.
            print(f"[scan_plantvillage] WARNING: skipping unrecognised folder '{raw_class}'")
            continue
        for img_path in list_images(class_dir, extensions):
            rows.append({
                "dataset": "plantvillage",
                "variant": variant_name,
                "raw_class": raw_class,
                "file_name": img_path.name,
                "filepath": str(img_path),
                "crop": meta["crop"],
                "disease": meta["disease"],
                "is_healthy": meta["is_healthy"],
                "joint_key": meta["joint_key"],
                "source_split": None,   # filled in by split.py from the official HF split
            })
    return pd.DataFrame(rows)


def scan_plantdoc(
    roots: dict[str, Path],
    taxonomy: Taxonomy,
    extensions: list[str],
) -> pd.DataFrame:
    rows = []
    for source_split, root in roots.items():   # {"train": Path, "test": Path}
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            raw_class = class_dir.name
            try:
                meta = taxonomy.lookup("plantdoc", raw_class)
            except KeyError:
                print(f"[scan_plantdoc] WARNING: skipping unrecognised folder '{raw_class}'")
                continue
            for img_path in list_images(class_dir, extensions):
                rows.append({
                    "dataset": "plantdoc",
                    "variant": "field",
                    "raw_class": raw_class,
                    "file_name": img_path.name,
                    "filepath": str(img_path),
                    "crop": meta["crop"],
                    "disease": meta["disease"],
                    "is_healthy": meta["is_healthy"],
                    "joint_key": meta["joint_key"],
                    "source_split": source_split,   # "train" or "test", as shipped by the dataset
                })
    return pd.DataFrame(rows)
