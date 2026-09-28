#!/usr/bin/env python3
"""Single entrypoint for the whole data stage. Run this first on Kaggle,
before touching any model code.

    python scripts/run_data_pipeline.py

Produces (all under --out, default data_manifests/):
    plantvillage_manifest.csv   one row per PlantVillage image, with split
    plantdoc_manifest.csv       one row per PlantDoc image, with split
    label_encoders.json         crop/disease/joint name <-> id, reused by every model
    dataset_stats.json          every count/measurement this run produced
    data_config_resolved.json   the exact config + resolved paths used for this run
    run_data_pipeline.summary.json / .log.jsonl   timestamped run log

Everything downstream (any model, any regime) reads only these files -- it
never re-touches /kaggle/input directly. That's what makes swapping in a new
model or regime "plug and play": the data contract is these CSVs + JSONs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import (
    resolve_plantvillage_roots, resolve_plantdoc_roots, require, DatasetNotFoundError,
)
from src.data.taxonomy import load_taxonomy, save_label_encoders
from src.data.scan import scan_plantvillage, scan_plantdoc
from src.data.split import attach_plantvillage_split, make_plantdoc_split
from src.data.stats import class_counts, image_size_sample, corrupt_check, channel_mean_std
from src.utils.io_utils import read_json, write_json, ensure_dir
from src.utils.seed_utils import set_global_seed
from src.utils.logging_utils import RunLogger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs_dir", default=str(REPO_ROOT / "configs"))
    p.add_argument("--data_meta_dir", default=None, help="Defaults to configs/paths.json's data_meta_dir")
    p.add_argument("--out", default=None, help="Defaults to configs/paths.json's manifests_out_dir")
    p.add_argument("--skip_corrupt_check", action="store_true")
    p.add_argument("--skip_channel_stats", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    configs_dir = Path(args.configs_dir)
    paths_cfg = read_json(configs_dir / "paths.json")
    data_cfg = read_json(configs_dir / "data_config.json")

    data_meta_dir = Path(args.data_meta_dir or paths_cfg["data_meta_dir"])
    out_dir = ensure_dir(args.out or paths_cfg["manifests_out_dir"])

    logger = RunLogger(out_dir, "run_data_pipeline")
    set_global_seed(data_cfg["seed"])
    logger.log("start", seed=data_cfg["seed"])

    taxonomy = load_taxonomy(configs_dir)
    logger.log("taxonomy_loaded",
               num_crops=taxonomy.num_crops(),
               num_diseases=taxonomy.num_diseases(),
               num_joint_classes=taxonomy.num_joint())
    save_label_encoders(taxonomy, out_dir / "label_encoders.json")

    # ---------------- resolve real dataset locations ----------------
    pv_variants = paths_cfg.get("plantvillage_variants_to_use", ["color"])
    pv_roots = resolve_plantvillage_roots(paths_cfg, taxonomy, pv_variants)
    require(pv_roots, pv_variants, "PlantVillage variant(s)",
            [paths_cfg["kaggle_input_dir"]] + paths_cfg["local_search_roots"]["plantvillage"])
    logger.log("plantvillage_roots_resolved", roots={k: str(v) for k, v in pv_roots.items()})

    pd_roots = resolve_plantdoc_roots(paths_cfg, taxonomy)
    require(pd_roots, ["train", "test"], "PlantDoc split(s)",
            [paths_cfg["kaggle_input_dir"]] + paths_cfg["local_search_roots"]["plantdoc"])
    logger.log("plantdoc_roots_resolved", roots={k: str(v) for k, v in pd_roots.items()})

    # ---------------- scan ----------------
    exts = data_cfg["image_extensions"]
    pv_df = scan_plantvillage(pv_roots[pv_variants[0]], taxonomy, exts, variant_name=pv_variants[0])
    logger.log("plantvillage_scanned", n_images=len(pv_df), n_classes=pv_df["raw_class"].nunique())

    pd_df = scan_plantdoc(pd_roots, taxonomy, exts)
    logger.log("plantdoc_scanned", n_images=len(pd_df), n_classes=pd_df["raw_class"].nunique())

    # ---------------- split ----------------
    pv_df, pv_split_report = attach_plantvillage_split(pv_df, data_meta_dir)
    logger.log("plantvillage_split_attached", **pv_split_report)
    if pv_split_report["rows_unmatched"] > 0:
        logger.log("plantvillage_split_WARNING",
                    detail="Some scanned files didn't match the precomputed leaf-grouped split "
                           "(new files since it was built, or a renamed mirror). "
                           "They are marked split=UNMATCHED and excluded by default consumers.")

    pd_df, pd_split_report = make_plantdoc_split(
        pd_df,
        val_fraction_of_train=data_cfg["plantdoc"]["val_fraction_of_official_train"],
        seed=data_cfg["seed"],
        hash_dedupe=data_cfg["plantdoc"]["hash_dedupe"],
    )
    logger.log("plantdoc_split_made", **pd_split_report)

    pv_manifest_path = out_dir / "plantvillage_manifest.csv"
    pd_manifest_path = out_dir / "plantdoc_manifest.csv"
    pv_df.to_csv(pv_manifest_path, index=False)
    pd_df.to_csv(pd_manifest_path, index=False)
    logger.log("manifests_written", plantvillage=str(pv_manifest_path), plantdoc=str(pd_manifest_path))

    # ---------------- stats ----------------
    stats_cfg = data_cfg["stats"]
    stats = {
        "plantvillage": {
            "class_counts": class_counts(pv_df),
            "image_size_sample": image_size_sample(pv_df, n=stats_cfg["image_size_sample_n"], seed=data_cfg["seed"]),
        },
        "plantdoc": {
            "class_counts": class_counts(pd_df),
            "image_size_sample": image_size_sample(pd_df, n=stats_cfg["image_size_sample_n"], seed=data_cfg["seed"]),
        },
    }
    if not args.skip_corrupt_check and stats_cfg["run_corrupt_check"]:
        stats["plantvillage"]["corrupt_files"] = corrupt_check(pv_df, sample_n=stats_cfg["corrupt_check_sample_n"])
        stats["plantdoc"]["corrupt_files"] = corrupt_check(pd_df, sample_n=stats_cfg["corrupt_check_sample_n"])
        logger.log("corrupt_check_done",
                   plantvillage_bad=len(stats["plantvillage"]["corrupt_files"]),
                   plantdoc_bad=len(stats["plantdoc"]["corrupt_files"]))
    if not args.skip_channel_stats:
        stats["plantvillage"]["channel_mean_std"] = channel_mean_std(pv_df, n=stats_cfg["channel_stats_sample_n"], seed=data_cfg["seed"])
        stats["plantdoc"]["channel_mean_std"] = channel_mean_std(pd_df, n=stats_cfg["channel_stats_sample_n"], seed=data_cfg["seed"])

    write_json(out_dir / "dataset_stats.json", stats)
    logger.log("stats_written", path=str(out_dir / "dataset_stats.json"))

    # ---------------- resolved config (every number this run used, in one file) ----------------
    resolved = {
        "seed": data_cfg["seed"],
        "plantvillage_roots_used": {k: str(v) for k, v in pv_roots.items()},
        "plantdoc_roots_used": {k: str(v) for k, v in pd_roots.items()},
        "plantvillage_split_report": pv_split_report,
        "plantdoc_split_report": pd_split_report,
        "num_crops": taxonomy.num_crops(),
        "num_diseases": taxonomy.num_diseases(),
        "num_joint_classes": taxonomy.num_joint(),
        "data_config": data_cfg,
        "paths_config": paths_cfg,
    }
    write_json(out_dir / "data_config_resolved.json", resolved)

    logger.write_summary(resolved)
    logger.log("done")
    logger.close()

    print("\n=== DATA PIPELINE COMPLETE ===")
    print(f"PlantVillage: {len(pv_df)} images -> {pv_manifest_path}")
    print(f"PlantDoc:     {len(pd_df)} images -> {pd_manifest_path}")
    print(f"Stats:        {out_dir / 'dataset_stats.json'}")
    print(f"Everything a model needs (label ids, split assignment, image paths) "
          f"is now in {out_dir}/ -- see assets/docs/03_data_pipeline_guide.md")


if __name__ == "__main__":
    try:
        main()
    except DatasetNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
