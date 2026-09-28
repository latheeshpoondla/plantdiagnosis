"""Small, dependency-light file I/O helpers used across the data pipeline.

Kept deliberately boring: every function here does exactly one thing so that
scan.py / split.py / stats.py / run_data_pipeline.py don't each reinvent
"read this json" or "hash this file".
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if missing. Returns it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def read_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def md5_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Exact-content hash. Used to catch true duplicate images so they never
    end up split across train/val/test (a cheap, real leakage guard)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Non-recursive listing of files in `directory` whose suffix is in
    `extensions` (case handled by the caller passing both cases, matching
    configs/data_config.json's image_extensions list).

    Deliberately does NOT call `Path.is_file()` on every entry: on a
    dataset with 50k+ files, that's 50k extra stat() syscalls, and on a
    network/cloud-synced mount (measured: this project's Kaggle-clone
    dev machine) each stat costs several milliseconds -- turning a
    sub-second directory scan into several minutes. A directory whose name
    happens to end in one of these image extensions is not a real-world
    concern for these datasets, so filtering on suffix alone is safe and
    ~100-200x faster in practice."""
    exts = set(extensions)
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in exts)
