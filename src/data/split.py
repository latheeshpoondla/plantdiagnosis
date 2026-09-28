"""Assigns a final split column ("train" / "val" / "test") to a scanned
manifest.

PlantVillage: reuses the leaf-grouped official split pre-computed into
data_meta/plantvillage/resolved_splits/color_{train,val,test}.csv (see
assets/docs/02_taxonomy_mapping.md for how those were built from the
HuggingFace mohanty/PlantVillage leaf-map). Matching is by
(raw_class, file_name) -- deliberately NOT by full path, since the path
prefix differs between the machine those CSVs were built on and wherever
Kaggle mounts the dataset for a given notebook run.

PlantDoc: the official train/ and test/ folders are trusted as the
train/test boundary; test is locked. A validation set is carved out of the
official train folder here, at run time, stratified by joint class, with an
exact-file-hash de-duplication pass so a literal duplicate image can't end
up straddling train and val.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.utils.io_utils import md5_of_file


def attach_plantvillage_split(df: pd.DataFrame, data_meta_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    data_meta_dir = Path(data_meta_dir) / "plantvillage" / "resolved_splits"
    split_frames = []
    for split_name in ("train", "val", "test"):
        p = data_meta_dir / f"color_{split_name}.csv"
        sdf = pd.read_csv(p)
        sdf["split"] = split_name
        split_frames.append(sdf[["class_name", "file_name", "split"]])
    lookup = pd.concat(split_frames, ignore_index=True).rename(columns={"class_name": "raw_class"})

    merged = df.merge(lookup, on=["raw_class", "file_name"], how="left")
    n_unmatched = int(merged["split"].isna().sum())
    if n_unmatched:
        merged["split"] = merged["split"].fillna("UNMATCHED")

    report = {
        "rows_total": len(merged),
        "rows_matched": len(merged) - n_unmatched,
        "rows_unmatched": n_unmatched,
        "match_rate": round(1 - n_unmatched / max(len(merged), 1), 4),
        "split_counts": merged["split"].value_counts().to_dict(),
    }
    return merged, report


def make_plantdoc_split(
    df: pd.DataFrame,
    val_fraction_of_train: float = 0.12,
    seed: int = 42,
    hash_dedupe: bool = True,
) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)
    df = df.copy()
    df["split"] = df["source_split"]  # "test" rows are locked as-is

    train_mask = df["source_split"] == "train"
    train_df = df.loc[train_mask]

    if hash_dedupe:
        hashes = {i: md5_of_file(fp) for i, fp in train_df["filepath"].items()}
        train_df = train_df.assign(_hash=pd.Series(hashes))
    else:
        train_df = train_df.assign(_hash=train_df.index.astype(str))
    # Plain dict, not a pandas Series -- Series.values is a ndarray *property*,
    # not a dict-like .values() method, and silently calling it blows up.
    dup_groups: dict[str, list[int]] = (
        train_df.groupby("_hash").apply(lambda g: list(g.index), include_groups=False).to_dict()
    )

    # group by joint_key (class) to keep val stratified; within each class,
    # allocate whole hash-groups (never split an exact duplicate across sets)
    group_class = {}
    for h, idxs in dup_groups.items():
        group_class[h] = train_df.loc[idxs[0], "joint_key"]

    by_class_groups: dict[str, list[str]] = {}
    for h, cls in group_class.items():
        by_class_groups.setdefault(cls, []).append(h)

    val_indices: list[int] = []
    for cls, hashes_in_class in by_class_groups.items():
        rng.shuffle(hashes_in_class)
        n_rows_cls = sum(len(dup_groups[h]) for h in hashes_in_class)
        target = round(n_rows_cls * val_fraction_of_train)
        acc = 0
        for h in hashes_in_class:
            if acc >= target:
                break
            val_indices.extend(dup_groups[h])
            acc += len(dup_groups[h])

    df.loc[val_indices, "split"] = "val"
    df.loc[train_mask & ~df.index.isin(val_indices), "split"] = "train"

    n_dupe_rows = sum(len(v) for v in dup_groups.values() if len(v) > 1)
    report = {
        "rows_total": len(df),
        "official_train_rows": int(train_mask.sum()),
        "official_test_rows": int((df["source_split"] == "test").sum()),
        "exact_duplicate_rows_found_in_train": n_dupe_rows,
        "unique_hash_groups_in_train": len(dup_groups),
        "resolved_train_rows": int((df["split"] == "train").sum()),
        "resolved_val_rows": int((df["split"] == "val").sum()),
        "resolved_test_rows": int((df["split"] == "test").sum()),
        "val_fraction_target": val_fraction_of_train,
        "seed": seed,
        "hash_dedupe": hash_dedupe,
    }
    return df, report
