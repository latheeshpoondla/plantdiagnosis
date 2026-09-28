#!/usr/bin/env python3
"""Builds a tiny fake PlantVillage + PlantDoc folder tree (real, openable
JPGs, a handful per class) so the whole pipeline can be smoke-tested in
seconds without the real ~2GB / ~1GB downloads. Not used in production --
this is a development/CI aid.

    python scripts/make_synthetic_fixture.py --out /tmp/pd_fixture --n_per_class 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image


def make_image(path: Path, seed: int):
    rng = random.Random(seed)
    im = Image.new("RGB", (64, 64))
    im.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(64 * 64)])
    im.save(path, quality=70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    pvd = json.loads((REPO_ROOT / "configs/taxonomy/plantvillage_classes.json").read_text())
    pdc = json.loads((REPO_ROOT / "configs/taxonomy/plantdoc_classes.json").read_text())

    i = 0
    for cls in pvd:
        d = out / "kaggle_input/plantvillage-dataset/color/color" / cls
        d.mkdir(parents=True, exist_ok=True)
        for k in range(args.n_per_class):
            i += 1
            make_image(d / f"img_{k}.jpg", seed=i)

    for split in ("train", "test"):
        for cls in pdc:
            d = out / "kaggle_input/plant-doc-dataset/PlantDoc-Dataset" / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for k in range(args.n_per_class):
                i += 1
                make_image(d / f"img_{k}.jpg", seed=i)

    print(f"Fixture built at {out} ({i} images)")


if __name__ == "__main__":
    main()
