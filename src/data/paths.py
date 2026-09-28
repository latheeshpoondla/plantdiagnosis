"""Finds the real dataset folders wherever they happen to be mounted.

Kaggle datasets are notoriously inconsistent about nesting (a dataset
uploaded as a folder called 'color' can end up mounted at
/kaggle/input/plantvillage-dataset/color/color/<class>/*.jpg because the
zip itself contained a 'color' folder). Rather than hardcoding a path and
breaking the moment Kaggle's packaging changes, this module walks each
search root and looks for a directory whose *immediate children* match the
known class-folder names for that dataset (from configs/taxonomy/). The
same function works on Kaggle (searching /kaggle/input) and locally
(searching configs/paths.json's local_search_roots), so the pipeline needs
no "am I on Kaggle" branch anywhere else in the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.data.taxonomy import Taxonomy


def _immediate_subdirs(path: Path) -> set[str]:
    try:
        return {p.name for p in path.iterdir() if p.is_dir()}
    except (PermissionError, OSError):
        return set()


def find_signature_dir(
    base: Path,
    expected_names: set[str],
    min_match_frac: float = 0.7,
    max_depth: int = 5,
) -> Optional[Path]:
    """BFS from `base` (inclusive) up to `max_depth` levels, looking for the
    shallowest directory whose child-directory names overlap `expected_names`
    by at least `min_match_frac`. Case-insensitive match on names."""
    if not base.is_dir():
        return None
    expected_lower = {n.lower() for n in expected_names}
    threshold = max(1, int(round(min_match_frac * len(expected_names))))

    frontier = [(base, 0)]
    seen = set()
    while frontier:
        current, depth = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)

        children = _immediate_subdirs(current)
        children_lower = {c.lower() for c in children}
        overlap = len(children_lower & expected_lower)
        if overlap >= threshold:
            return current

        if depth < max_depth:
            try:
                for child in sorted(current.iterdir()):
                    if child.is_dir() and not child.name.startswith("."):
                        frontier.append((child, depth + 1))
            except (PermissionError, OSError):
                pass
    return None


def _search_roots(base_dirs: list[str]) -> list[Path]:
    return [Path(b) for b in base_dirs if Path(b).exists()]


def resolve_plantvillage_roots(
    paths_cfg: dict,
    taxonomy: Taxonomy,
    variants: Optional[list[str]] = None,
) -> dict[str, Path]:
    """Returns {variant_name: resolved_path}, e.g. {'color': Path(...)}.
    A variant missing from the result simply wasn't found in any search root."""
    variants = variants or paths_cfg.get("plantvillage_variants_to_use", ["color"])
    expected = set(taxonomy.pvd_map.keys())  # the 38 raw class-folder names

    bases: list[str] = []
    kaggle_input = paths_cfg.get("kaggle_input_dir", "/kaggle/input")
    slug = paths_cfg.get("kaggle_dataset_slugs", {}).get("plantvillage")
    if slug:
        bases.append(os.path.join(kaggle_input, slug))
    bases.extend(paths_cfg.get("local_search_roots", {}).get("plantvillage", []))

    found: dict[str, Path] = {}
    for base in _search_roots(bases):
        for variant in variants:
            if variant in found:
                continue
            # First try: a subdir literally named after the variant (color/grayscale/segmented)
            candidate = base / variant
            hit = find_signature_dir(candidate, expected, max_depth=3) if candidate.exists() else None
            if hit is None:
                # Fall back to a full signature search under this base (handles
                # layouts where the variant folder isn't obviously named).
                hit = find_signature_dir(base, expected, max_depth=5)
            if hit is not None:
                found[variant] = hit
    return found


def resolve_plantdoc_roots(paths_cfg: dict, taxonomy: Taxonomy) -> dict[str, Path]:
    """Returns {'train': Path, 'test': Path} for whichever were found."""
    expected = set(taxonomy.pd_map.keys())  # the 27 raw class-folder names

    bases: list[str] = []
    kaggle_input = paths_cfg.get("kaggle_input_dir", "/kaggle/input")
    slug = paths_cfg.get("kaggle_dataset_slugs", {}).get("plantdoc")
    if slug:
        bases.append(os.path.join(kaggle_input, slug))
    bases.extend(paths_cfg.get("local_search_roots", {}).get("plantdoc", []))

    found: dict[str, Path] = {}
    for base in _search_roots(bases):
        for split_name in ("train", "test"):
            if split_name in found:
                continue
            # Look for a dir literally named train/test anywhere under base,
            # then confirm its children match the 27-class signature.
            for candidate in [p for p in base.rglob("*")
                               if p.is_dir() and p.name.lower() == split_name]:
                if len(_immediate_subdirs(candidate) & {n.lower() for n in expected}) or \
                   find_signature_dir(candidate, expected, max_depth=1) == candidate:
                    found[split_name] = candidate
                    break
    return found


class DatasetNotFoundError(RuntimeError):
    pass


def require(found: dict, needed: list[str], dataset_label: str, searched: list[str]) -> None:
    missing = [k for k in needed if k not in found]
    if missing:
        raise DatasetNotFoundError(
            f"Could not locate {dataset_label} {missing} under any of: {searched}. "
            f"Check configs/paths.json's kaggle_dataset_slugs / local_search_roots, "
            f"and that the dataset is attached to this Kaggle notebook."
        )
