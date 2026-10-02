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

    def crop_healthy_diseased_coverage(self) -> dict:
        """For every crop, whether a healthy class and/or any diseased
        class(es) exist in each dataset's raw taxonomy. This is a property
        of the LABEL DEFINITIONS (configs/taxonomy/*.json), independent of
        which images actually got scanned -- e.g. it will say PlantDoc has
        no healthy corn class even before any image is ever loaded, because
        no such raw folder exists in the dataset's design.

        Answers "does this crop have both a healthy and a diseased class in
        each dataset, or is it healthy-only / diseased-only / absent?" --
        see assets/docs/02_taxonomy_mapping.md section 6 for the full table
        and what it means for each experiment regime.
        """
        def per_dataset(raw_map: dict) -> dict:
            cov: dict[str, dict] = {}
            for meta in raw_map.values():
                c = meta["crop"]
                entry = cov.setdefault(c, {"healthy": False, "diseases": set()})
                if meta["is_healthy"]:
                    entry["healthy"] = True
                else:
                    entry["diseases"].add(meta["disease"])
            return cov

        pvd_cov = per_dataset(self.pvd_map)
        pd_cov = per_dataset(self.pd_map)
        all_crops = sorted(set(pvd_cov) | set(pd_cov))

        report = {}
        for c in all_crops:
            pv = pvd_cov.get(c, {"healthy": False, "diseases": set()})
            pdc = pd_cov.get(c, {"healthy": False, "diseases": set()})
            flags = []
            if c not in pd_cov:
                flags.append("absent_from_plantdoc")
            else:
                if pv["healthy"] and not pdc["healthy"]:
                    flags.append("no_healthy_example_in_plantdoc")
                if pv["diseases"] and not pdc["diseases"]:
                    flags.append("no_diseased_example_in_plantdoc")
            if not pv["healthy"]:
                flags.append("no_healthy_class_in_plantvillage")
            if not pv["diseases"]:
                flags.append("no_diseased_class_in_plantvillage")
            report[c] = {
                "plantvillage_healthy": pv["healthy"],
                "plantvillage_n_diseases": len(pv["diseases"]),
                "plantdoc_healthy": pdc["healthy"],
                "plantdoc_n_diseases": len(pdc["diseases"]),
                "flags": flags,
            }
        return report


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
