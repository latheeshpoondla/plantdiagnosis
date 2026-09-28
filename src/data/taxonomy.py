"""Loads configs/taxonomy/*.json and turns them into fast lookup tables +
deterministic label encoders.

The three JSON files are generated data (see
assets/docs/02_taxonomy_mapping.md for how and why), not something this
module invents -- it only loads and indexes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.io_utils import read_json


@dataclass
class Taxonomy:
    pvd_map: dict          # raw PlantVillage folder name -> {crop, disease, is_healthy, joint_key}
    pd_map: dict            # raw PlantDoc folder name -> {crop, disease, is_healthy, assumption, joint_key}
    canonical: dict          # crops / diseases / joint-class set comparisons

    crop_to_id: dict = field(default_factory=dict)
    id_to_crop: dict = field(default_factory=dict)
    disease_to_id: dict = field(default_factory=dict)
    id_to_disease: dict = field(default_factory=dict)
    joint_to_id: dict = field(default_factory=dict)
    id_to_joint: dict = field(default_factory=dict)

    def __post_init__(self):
        crops = sorted(self.canonical["crops"])
        diseases = sorted(self.canonical["diseases"])
        # Union of every joint key seen in either dataset -- both PlantVillage-only
        # classes and PlantDoc classes get a stable id, so a model trained on PVD
        # joint labels and evaluated/fine-tuned on PlantDoc never has to remap ids.
        joints = sorted(set(self.canonical["joint_classes_plantvillage"])
                         | set(self.canonical["joint_classes_plantdoc"]))
        self.crop_to_id = {c: i for i, c in enumerate(crops)}
        self.id_to_crop = {i: c for c, i in self.crop_to_id.items()}
        self.disease_to_id = {d: i for i, d in enumerate(diseases)}
        self.id_to_disease = {i: d for d, i in self.disease_to_id.items()}
        self.joint_to_id = {j: i for i, j in enumerate(joints)}
        self.id_to_joint = {i: j for j, i in self.joint_to_id.items()}

    def lookup(self, dataset: str, raw_class_name: str) -> dict:
        """dataset: 'plantvillage' or 'plantdoc'. Returns {crop, disease, is_healthy, joint_key}."""
        table = self.pvd_map if dataset == "plantvillage" else self.pd_map
        if raw_class_name not in table:
            raise KeyError(
                f"Unknown raw class '{raw_class_name}' for dataset '{dataset}'. "
                f"The taxonomy json in configs/taxonomy/ needs updating if this "
                f"is a genuinely new class folder."
            )
        return table[raw_class_name]

    def num_crops(self) -> int:
        return len(self.crop_to_id)

    def num_diseases(self) -> int:
        return len(self.disease_to_id)

    def num_joint(self) -> int:
        return len(self.joint_to_id)


def load_taxonomy(configs_dir: str | Path = "configs") -> Taxonomy:
    tdir = Path(configs_dir) / "taxonomy"
    pvd_map = read_json(tdir / "plantvillage_classes.json")
    pd_map = read_json(tdir / "plantdoc_classes.json")
    canonical = read_json(tdir / "canonical_taxonomy.json")
    return Taxonomy(pvd_map=pvd_map, pd_map=pd_map, canonical=canonical)


def save_label_encoders(taxonomy: Taxonomy, out_path: str | Path) -> None:
    from src.utils.io_utils import write_json

    write_json(out_path, {
        "crop_to_id": taxonomy.crop_to_id,
        "disease_to_id": taxonomy.disease_to_id,
        "joint_to_id": taxonomy.joint_to_id,
        "num_crops": taxonomy.num_crops(),
        "num_diseases": taxonomy.num_diseases(),
        "num_joint": taxonomy.num_joint(),
    })
