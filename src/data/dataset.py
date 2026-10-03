"""The torch Dataset used by every model formulation described in
assets/PlantDiagnosis_Plan.md / the project's experiment design: crop-only,
disease-only, joint-label, and shared-encoder multitask all read the same
manifest CSV and just ask for a different `label_mode`.

The torch import is optional at import time (falls back to a plain object
base class) so this file can be imported and its non-torch logic exercised
on a machine without torch installed; on Kaggle, torch is always present and
PlantLeafDataset behaves as a normal torch.utils.data.Dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image

try:
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised on non-Kaggle machines only
    _TorchDataset = object
    _HAS_TORCH = False

LabelMode = Literal["crop", "disease", "joint", "multitask"]


class PlantLeafDataset(_TorchDataset):
    def __init__(
        self,
        manifest: "pd.DataFrame | str | Path",
        split: str,
        label_mode: LabelMode,
        taxonomy,
        transform=None,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "torch is required to use PlantLeafDataset (it isn't installed "
                "in this environment). This is expected outside Kaggle."
            )
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        # "all" -- every row regardless of split -- exists for cross-dataset
        # zero-shot eval (e.g. regime 1's PlantDoc eval in scripts/train.py):
        # when a dataset was never trained/validated on, its train/val splits
        # aren't held out from anything and are fair to evaluate on too, and
        # using all of them maximizes N for a dataset as thin per-class as
        # PlantDoc (see assets/docs/05_modelling_decisions.md).
        self.df = df.reset_index(drop=True) if split == "all" else df.loc[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split!r} in the given manifest.")
        self.label_mode = label_mode
        self.taxonomy = taxonomy
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _labels_for_row(self, row) -> int | tuple[int, int]:
        if self.label_mode == "crop":
            return self.taxonomy.crop_to_id[row["crop"]]
        if self.label_mode == "disease":
            return self.taxonomy.disease_to_id[row["disease"]]
        if self.label_mode == "joint":
            return self.taxonomy.joint_to_id[row["joint_key"]]
        if self.label_mode == "multitask":
            return (
                self.taxonomy.crop_to_id[row["crop"]],
                self.taxonomy.disease_to_id[row["disease"]],
            )
        raise ValueError(f"Unknown label_mode {self.label_mode!r}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self._labels_for_row(row)
        return image, label
