"""Exploratory data analysis for the Plant Diagnosis project.

Reads the OUTPUT of scripts/run_data_pipeline.py (data_manifests/*.csv/*.json)
-- it never touches /kaggle/input itself. Run the data pipeline first.

Designed for one function call per Kaggle cell: each plot function loads
nothing on its own, takes the already-loaded dataframes/stats as arguments,
prints a short description of what it found, displays the figure inline
(via plt.show() -- works whether called one-per-cell or all at once through
run_full_eda()), saves a PNG with a clear filename, and returns the saved
path. Call `load_eda_inputs()` once at the top of the notebook, then call
whichever plot functions you want, in any order, in their own cells.
`run_full_eda()` runs all of them in sequence -- and shows every figure
inline, one after another -- if you just want everything.

Typical Kaggle usage
---------------------
    # cell 1
    from src.eda import *
    pv_df, pd_df, stats, taxonomy, label_encoders, resolved_cfg = load_eda_inputs()

    # cell 2
    plot_dataset_split_overview(pv_df, pd_df)

    # cell 3
    plot_joint_class_counts(pv_df, "plantvillage", taxonomy)

    # cell 4
    plot_joint_class_counts(pd_df, "plantdoc", taxonomy)

    # ... etc, one plot per cell, OR:

    # one cell, everything:
    run_full_eda()

All plots are saved under EDA_OUT_DIR (default "eda_outputs/"), numbered so
they sort in the order a human would want to read them.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EDA_OUT_DIR = "eda_outputs"

# ---------------------------------------------------------------------------
# Palette -- validated categorical (CVD-safe adjacent pairs), sequential blue
# ramp, and status colors, per the project's dataviz convention. Fixed order,
# never cycled per-call, so "plantvillage" and "train" always mean the same
# color across every figure in this file.
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS_GOOD = "#0ca30c"       # healthy
STATUS_CRITICAL = "#d03b3b"   # diseased

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

DATASET_COLOR = {"plantvillage": CATEGORICAL[0], "plantdoc": CATEGORICAL[1]}
DATASET_LABEL = {"plantvillage": "PlantVillage", "plantdoc": "PlantDoc"}
SPLIT_COLOR = {"train": CATEGORICAL[0], "val": CATEGORICAL[3], "test": CATEGORICAL[2]}
SPLIT_ORDER = ["train", "val", "test"]


def set_eda_style() -> None:
    """Applies the project's plot style. Called automatically on import;
    call again after any `plt.rcdefaults()` to restore it."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": True,
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "legend.frameon": False,
    })


set_eda_style()


# ---------------------------------------------------------------------------
# small internal helpers
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _savefig(fig, out_dir: str | Path, filename: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path)
    print(f"Saved -> {path}")
    # Display inline (Jupyter/Kaggle's inline backend renders + auto-closes on show()),
    # then close explicitly too so a non-inline backend doesn't leak figures across the
    # 15-19 plots run_full_eda() produces in one cell.
    plt.show()
    plt.close(fig)
    return path


def _sample_filepaths(df: pd.DataFrame, n: int, seed: int) -> pd.Series:
    return df["filepath"].sample(min(n, len(df)), random_state=seed)


# ---------------------------------------------------------------------------
# loader -- call once per notebook session
# ---------------------------------------------------------------------------
def load_eda_inputs(manifests_dir: str | Path = "data_manifests", configs_dir: str | Path = "configs"):
    """Loads everything scripts/run_data_pipeline.py produced. Returns
    (pv_df, pd_df, stats, taxonomy, label_encoders, resolved_cfg).

    Raises FileNotFoundError with an actionable message if the pipeline
    hasn't been run yet -- this module never scans /kaggle/input itself.
    """
    from src.data.taxonomy import load_taxonomy

    manifests_dir = Path(manifests_dir)
    pv_path = manifests_dir / "plantvillage_manifest.csv"
    pd_path = manifests_dir / "plantdoc_manifest.csv"
    stats_path = manifests_dir / "dataset_stats.json"
    label_path = manifests_dir / "label_encoders.json"
    resolved_path = manifests_dir / "data_config_resolved.json"

    missing = [p for p in (pv_path, pd_path, stats_path, label_path, resolved_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing pipeline output(s): {[str(p) for p in missing]}. "
            f"Run `python scripts/run_data_pipeline.py` first -- EDA reads its "
            f"output under {manifests_dir}/, it does not scan /kaggle/input itself."
        )

    pv_df = pd.read_csv(pv_path)
    pd_df = pd.read_csv(pd_path)
    stats = json.loads(stats_path.read_text())
    label_encoders = json.loads(label_path.read_text())
    resolved_cfg = json.loads(resolved_path.read_text())
    taxonomy = load_taxonomy(configs_dir)

    _section("Loaded data pipeline output")
    print(f"PlantVillage manifest: {len(pv_df):,} rows  |  splits: {pv_df['split'].value_counts().to_dict()}")
    print(f"PlantDoc manifest:     {len(pd_df):,} rows  |  splits: {pd_df['split'].value_counts().to_dict()}")
    print(f"Taxonomy: {taxonomy.num_crops()} crops, {taxonomy.num_diseases()} diseases, "
          f"{taxonomy.num_joint()} joint classes (shared id space across both datasets)")
    n_unmatched = int((pv_df["split"] == "UNMATCHED").sum())
    if n_unmatched:
        print(f"WARNING: {n_unmatched} PlantVillage rows didn't match the precomputed leaf-grouped "
              f"split (split='UNMATCHED') -- see assets/docs/02_taxonomy_mapping.md.")
    return pv_df, pd_df, stats, taxonomy, label_encoders, resolved_cfg


# ---------------------------------------------------------------------------
# 1. dataset / split overview
# ---------------------------------------------------------------------------
def plot_dataset_split_overview(pv_df: pd.DataFrame, pd_df: pd.DataFrame, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Two panels: absolute image counts per split per dataset, and the same
    as split proportions -- the first "how big / how balanced is each
    dataset's split" look."""
    _section("1. Dataset & split overview")
    dfs = {"plantvillage": pv_df, "plantdoc": pd_df}
    counts = {ds: df["split"].value_counts().reindex(SPLIT_ORDER).fillna(0).astype(int) for ds, df in dfs.items()}
    for ds, c in counts.items():
        total = c.sum()
        print(f"{DATASET_LABEL[ds]}: {total:,} images total | " +
              " / ".join(f"{s}={c[s]:,} ({100*c[s]/total:.1f}%)" for s in SPLIT_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(SPLIT_ORDER))
    width = 0.35
    for i, ds in enumerate(dfs):
        axes[0].bar(x + (i - 0.5) * width, counts[ds].values, width,
                    label=DATASET_LABEL[ds], color=DATASET_COLOR[ds])
    axes[0].set_xticks(x); axes[0].set_xticklabels([s.capitalize() for s in SPLIT_ORDER])
    axes[0].set_ylabel("images"); axes[0].set_title("Images per split (count)")
    axes[0].legend()

    bottoms = {ds: 0 for ds in dfs}
    y = np.arange(len(dfs))
    for s_i, split in enumerate(SPLIT_ORDER):
        vals = [100 * counts[ds][split] / counts[ds].sum() for ds in dfs]
        axes[1].barh(y, vals, left=[bottoms[ds] for ds in dfs], color=SPLIT_COLOR[split],
                     label=split.capitalize(), height=0.5)
        for ds, v in zip(dfs, vals):
            bottoms[ds] += v
    axes[1].set_yticks(y); axes[1].set_yticklabels([DATASET_LABEL[ds] for ds in dfs])
    axes[1].set_xlabel("% of dataset"); axes[1].set_title("Split proportions")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    axes[1].set_xlim(0, 100)

    fig.suptitle("Dataset & split overview", y=1.02, fontsize=13, fontweight="bold")
    return _savefig(fig, out_dir, "01_dataset_split_overview.png")


# ---------------------------------------------------------------------------
# 2. joint class counts (per dataset -- different class sets/sizes)
# ---------------------------------------------------------------------------
def plot_joint_class_counts(df: pd.DataFrame, dataset_name: str, taxonomy, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Horizontal stacked bar (by split) of every joint crop__disease class
    for ONE dataset, sorted by total count -- the main "what does the class
    list actually look like" plot. Call once per dataset."""
    _section(f"2. Joint class counts -- {DATASET_LABEL[dataset_name]}")
    pivot = df.pivot_table(index="joint_key", columns="split", values="filepath",
                            aggfunc="count", fill_value=0)
    for s in SPLIT_ORDER:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[SPLIT_ORDER]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total")
    n_classes = len(pivot)
    print(f"{n_classes} joint classes present in this manifest. "
          f"Largest: {pivot['total'].idxmax()} (n={pivot['total'].max()}). "
          f"Smallest: {pivot['total'].idxmin()} (n={pivot['total'].min()}).")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * n_classes)))
    left = np.zeros(n_classes)
    y = np.arange(n_classes)
    for split in SPLIT_ORDER:
        ax.barh(y, pivot[split].values, left=left, color=SPLIT_COLOR[split], label=split.capitalize(), height=0.7)
        left += pivot[split].values
    ax.set_yticks(y); ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("images"); ax.set_title(f"Joint class counts by split -- {DATASET_LABEL[dataset_name]}")
    ax.legend(loc="lower right")
    return _savefig(fig, out_dir, f"02_joint_class_counts_{dataset_name}.png")


# ---------------------------------------------------------------------------
# 3. crop-level counts, both datasets side by side
# ---------------------------------------------------------------------------
def plot_crop_counts(pv_df: pd.DataFrame, pd_df: pd.DataFrame, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """One combined horizontal grouped bar of crop totals, PlantVillage vs
    PlantDoc, log-scaled x-axis (PlantVillage is roughly an order of
    magnitude larger per crop) -- the main cross-dataset scale comparison."""
    _section("3. Crop-level counts -- PlantVillage vs PlantDoc")
    pv_c = pv_df["crop"].value_counts()
    pd_c = pd_df["crop"].value_counts()
    crops = sorted(set(pv_c.index) | set(pd_c.index), key=lambda c: -pv_c.get(c, 0))
    pv_vals = [pv_c.get(c, 0) for c in crops]
    pd_vals = [pd_c.get(c, 0) for c in crops]
    missing_in_pd = [c for c in crops if pd_c.get(c, 0) == 0]
    print(f"{len(crops)} crops total. Crops with zero PlantDoc images: {missing_in_pd or 'none'}.")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(crops))))
    y = np.arange(len(crops))
    height = 0.38
    ax.barh(y + height / 2, pv_vals, height, color=DATASET_COLOR["plantvillage"], label="PlantVillage")
    ax.barh(y - height / 2, pd_vals, height, color=DATASET_COLOR["plantdoc"], label="PlantDoc")
    ax.set_yticks(y); ax.set_yticklabels(crops)
    ax.set_xscale("log")
    ax.set_xlabel("images (log scale)"); ax.set_title("Crop-level image counts (PlantVillage vs PlantDoc)")
    ax.legend()
    return _savefig(fig, out_dir, "03_crop_counts_comparison.png")


# ---------------------------------------------------------------------------
# 4. disease-level counts (per dataset)
# ---------------------------------------------------------------------------
def plot_disease_counts(df: pd.DataFrame, dataset_name: str, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Horizontal bar of disease-level totals for one dataset (diseases are
    merged across crops where the condition is the same -- see
    assets/docs/02_taxonomy_mapping.md -- so 'healthy' sums every crop)."""
    _section(f"4. Disease-level counts -- {DATASET_LABEL[dataset_name]}")
    counts = df["disease"].value_counts().sort_values()
    print(f"{len(counts)} distinct disease labels. Most common: {counts.idxmax()} (n={counts.max()}). "
          f"Rarest: {counts.idxmin()} (n={counts.min()}).")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(counts))))
    colors = [STATUS_GOOD if d == "healthy" else DATASET_COLOR[dataset_name] for d in counts.index]
    ax.barh(np.arange(len(counts)), counts.values, color=colors)
    ax.set_yticks(np.arange(len(counts))); ax.set_yticklabels(counts.index)
    ax.set_xscale("log")
    ax.set_xlabel("images (log scale)"); ax.set_title(f"Disease-level counts -- {DATASET_LABEL[dataset_name]}\n(green = healthy)")
    return _savefig(fig, out_dir, f"04_disease_counts_{dataset_name}.png")


# ---------------------------------------------------------------------------
# 5. healthy vs diseased
# ---------------------------------------------------------------------------
def plot_healthy_vs_diseased(pv_df: pd.DataFrame, pd_df: pd.DataFrame, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Grouped bar: healthy vs diseased image counts, both datasets --
    status colors (green=healthy, red=diseased) since this is a state, not
    an identity, per the project's color convention."""
    _section("5. Healthy vs diseased")
    rows = []
    for name, df in [("plantvillage", pv_df), ("plantdoc", pd_df)]:
        healthy = int(df["is_healthy"].sum())
        diseased = int((~df["is_healthy"]).sum())
        rows.append((name, healthy, diseased))
        print(f"{DATASET_LABEL[name]}: {healthy:,} healthy ({100*healthy/len(df):.1f}%), "
              f"{diseased:,} diseased ({100*diseased/len(df):.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(rows))
    width = 0.35
    ax.bar(x - width / 2, [r[1] for r in rows], width, label="Healthy", color=STATUS_GOOD)
    ax.bar(x + width / 2, [r[2] for r in rows], width, label="Diseased", color=STATUS_CRITICAL)
    ax.set_xticks(x); ax.set_xticklabels([DATASET_LABEL[r[0]] for r in rows])
    ax.set_ylabel("images"); ax.set_title("Healthy vs diseased images")
    ax.legend()
    return _savefig(fig, out_dir, "05_healthy_vs_diseased.png")


# ---------------------------------------------------------------------------
# 6. class imbalance
# ---------------------------------------------------------------------------
def plot_class_imbalance(df: pd.DataFrame, dataset_name: str, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Sorted bar of joint-class totals (log y) with the imbalance ratio
    (largest/smallest class) printed -- the number that matters for deciding
    whether to use class weighting / oversampling."""
    _section(f"6. Class imbalance -- {DATASET_LABEL[dataset_name]}")
    counts = df["joint_key"].value_counts().sort_values(ascending=False)
    ratio = counts.max() / counts.min()
    print(f"Largest class: {counts.idxmax()} (n={counts.max()}) | Smallest: {counts.idxmin()} (n={counts.min()}) | "
          f"imbalance ratio (max/min) = {ratio:.1f}x")
    print(f"Median class size: {int(counts.median())} | classes below 50 images: {int((counts < 50).sum())}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(counts)), counts.values, color=DATASET_COLOR[dataset_name], width=0.8)
    ax.set_yscale("log")
    ax.set_xticks([]); ax.set_xlabel(f"{len(counts)} joint classes, sorted by size")
    ax.set_ylabel("images (log scale)")
    ax.set_title(f"Class imbalance -- {DATASET_LABEL[dataset_name]} (ratio {ratio:.1f}x)")
    return _savefig(fig, out_dir, f"06_class_imbalance_{dataset_name}.png")


# ---------------------------------------------------------------------------
# 7. crop coverage heatmap (taxonomy-level, see 02_taxonomy_mapping.md §6)
# ---------------------------------------------------------------------------
def plot_crop_coverage_heatmap(stats: dict, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Heatmap of crop x {PVD healthy, PVD #diseases, PD healthy, PD
    #diseases} from dataset_stats.json's taxonomy_crop_healthy_diseased_coverage
    -- the single plot that answers 'which crops have a healthy/diseased
    blind spot in which dataset' (see assets/docs/02_taxonomy_mapping.md §6)."""
    _section("7. Crop healthy/diseased coverage")
    cov = stats["taxonomy_crop_healthy_diseased_coverage"]
    crops = sorted(cov)
    cols = ["plantvillage_healthy", "plantvillage_n_diseases", "plantdoc_healthy", "plantdoc_n_diseases"]
    col_labels = ["PVD healthy?", "PVD #diseases", "PD healthy?", "PD #diseases"]
    mat = np.zeros((len(crops), len(cols)))
    for i, c in enumerate(crops):
        for j, col in enumerate(cols):
            v = cov[c][col]
            mat[i, j] = (1 if v else 0) if isinstance(v, bool) else v
    flagged = [c for c in crops if cov[c]["flags"]]
    print(f"{len(flagged)}/{len(crops)} crops have at least one coverage flag "
          f"(no healthy or no diseased example in one dataset): {flagged}")

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(crops))))
    # normalize each column independently (booleans vs disease-counts aren't comparable scales)
    norm_mat = mat / (mat.max(axis=0, keepdims=True) + 1e-9)
    im = ax.imshow(norm_mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(crops))); ax.set_yticklabels(crops)
    for i in range(len(crops)):
        for j in range(len(cols)):
            val = mat[i, j]
            text = ("yes" if val else "no") if j in (0, 2) else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=INK_PRIMARY if norm_mat[i, j] < 0.6 else "white")
    ax.set_title("Crop coverage: healthy & disease classes per dataset")
    return _savefig(fig, out_dir, "07_crop_coverage_heatmap.png")


# ---------------------------------------------------------------------------
# 8. joint class overlap between datasets
# ---------------------------------------------------------------------------
def plot_joint_class_overlap(taxonomy, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Bar of matched / PlantVillage-only / PlantDoc-only joint classes --
    how much of PlantVillage's label space PlantDoc actually exercises."""
    _section("8. Joint class overlap")
    matched = set(taxonomy.canonical["joint_classes_matched_both"])
    pv_only = set(taxonomy.canonical["joint_classes_plantvillage_only"])
    pd_only = set(taxonomy.canonical.get("joint_classes_plantdoc_only", []))
    print(f"Matched (in both): {len(matched)} | PlantVillage-only: {len(pv_only)} | PlantDoc-only: {len(pd_only)}")
    if pv_only:
        print("PlantVillage-only classes:", sorted(pv_only))

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Matched\n(both datasets)", "PlantVillage-only", "PlantDoc-only"]
    vals = [len(matched), len(pv_only), len(pd_only)]
    colors = [CATEGORICAL[2], CATEGORICAL[0], CATEGORICAL[1]]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("joint classes"); ax.set_title("Joint class overlap between datasets")
    return _savefig(fig, out_dir, "08_joint_class_overlap.png")


# ---------------------------------------------------------------------------
# 9. zero-support classes per split
# ---------------------------------------------------------------------------
def plot_zero_support_classes(stats: dict, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Grouped bar: how many joint classes have ZERO examples in each split
    of each dataset. Direct visualization of dataset_stats.json's
    zero_support_joint_classes (see assets/docs/02_taxonomy_mapping.md §6)."""
    _section("9. Zero-support classes per split")
    rows = []
    for ds in ("plantvillage", "plantdoc"):
        zsc = stats[ds]["zero_support_joint_classes"]
        for split in SPLIT_ORDER:
            n = len(zsc.get(split, []))
            rows.append((ds, split, n))
            if n:
                print(f"{DATASET_LABEL[ds]} / {split}: {n} classes with ZERO examples -> {zsc[split]}")
    if not any(r[2] for r in rows):
        print("No zero-support classes found in any split (unexpected for PlantDoc -- check the manifest).")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(SPLIT_ORDER))
    width = 0.35
    for i, ds in enumerate(("plantvillage", "plantdoc")):
        vals = [n for d, s, n in rows if d == ds]
        ax.bar(x + (i - 0.5) * width, vals, width, label=DATASET_LABEL[ds], color=DATASET_COLOR[ds])
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in SPLIT_ORDER])
    ax.set_ylabel("# joint classes with 0 examples")
    ax.set_title("Zero-support classes per split")
    ax.legend()
    return _savefig(fig, out_dir, "09_zero_support_classes.png")


# ---------------------------------------------------------------------------
# 10. image size distribution (real sampled images)
# ---------------------------------------------------------------------------
def plot_image_size_distribution(pv_df: pd.DataFrame, pd_df: pd.DataFrame, out_dir: str | Path = EDA_OUT_DIR,
                                  n_sample: int = 300, seed: int = 42) -> Path:
    """Scatter of width vs height for a real sample of images from each
    dataset, same axes, two series (never dual-axis) -- shows whether
    PlantVillage's uniform pre-resizing (256x256, see 01_data_analysis.md)
    still holds and how much PlantDoc's field photos vary."""
    _section("10. Image size distribution (sampled)")

    def sample_sizes(df, n):
        sizes, failed = [], 0
        for fp in _sample_filepaths(df, n, seed):
            try:
                with Image.open(fp) as im:
                    sizes.append(im.size)
            except (UnidentifiedImageError, OSError):
                failed += 1
        return np.array(sizes), failed

    pv_sizes, pv_failed = sample_sizes(pv_df, n_sample)
    pd_sizes, pd_failed = sample_sizes(pd_df, n_sample)
    for name, sizes, failed in [("PlantVillage", pv_sizes, pv_failed), ("PlantDoc", pd_sizes, pd_failed)]:
        if len(sizes):
            print(f"{name}: sampled {len(sizes)} images (unreadable: {failed}) | "
                  f"width {sizes[:,0].min()}-{sizes[:,0].max()} (mean {sizes[:,0].mean():.0f}) | "
                  f"height {sizes[:,1].min()}-{sizes[:,1].max()} (mean {sizes[:,1].mean():.0f})")

    fig, ax = plt.subplots(figsize=(7, 6))
    if len(pv_sizes):
        ax.scatter(pv_sizes[:, 0], pv_sizes[:, 1], s=14, alpha=0.5, color=DATASET_COLOR["plantvillage"], label="PlantVillage")
    if len(pd_sizes):
        ax.scatter(pd_sizes[:, 0], pd_sizes[:, 1], s=14, alpha=0.5, color=DATASET_COLOR["plantdoc"], label="PlantDoc")
    ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)")
    ax.set_title(f"Image size distribution (n={n_sample}/dataset sample)")
    ax.legend()
    return _savefig(fig, out_dir, "10_image_size_distribution.png")


# ---------------------------------------------------------------------------
# 11. channel value distribution (real sampled images)
# ---------------------------------------------------------------------------
def plot_channel_value_distribution(pv_df: pd.DataFrame, pd_df: pd.DataFrame, out_dir: str | Path = EDA_OUT_DIR,
                                     n_sample: int = 150, seed: int = 42) -> Path:
    """R/G/B pixel-value histograms (3 small multiples), PlantVillage vs
    PlantDoc overlaid in each -- a real distribution, not just the
    mean/std summary already in dataset_stats.json's channel_mean_std."""
    _section("11. Channel (RGB) value distribution (sampled)")

    def sample_pixels(df, n):
        chans = [[], [], []]
        for fp in _sample_filepaths(df, n, seed):
            try:
                with Image.open(fp) as im:
                    arr = np.asarray(im.convert("RGB"))
            except (UnidentifiedImageError, OSError):
                continue
            for c in range(3):
                chans[c].append(arr[:, :, c].ravel()[::37])  # stride-subsample, keeps memory sane
        return [np.concatenate(c) if c else np.array([]) for c in chans]

    pv_rgb = sample_pixels(pv_df, n_sample)
    pd_rgb = sample_pixels(pd_df, n_sample)
    for name, rgb in [("PlantVillage", pv_rgb), ("PlantDoc", pd_rgb)]:
        if len(rgb[0]):
            print(f"{name}: mean R/G/B = {rgb[0].mean():.1f}/{rgb[1].mean():.1f}/{rgb[2].mean():.1f} "
                  f"(0-255 scale, {sum(len(c) for c in rgb):,} sampled pixel-values)")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for i, (chan_name, color) in enumerate(zip(["Red", "Green", "Blue"], ["#d03b3b", "#0ca30c", "#2a78d6"])):
        if len(pv_rgb[i]):
            axes[i].hist(pv_rgb[i], bins=40, range=(0, 255), alpha=0.5, density=True,
                         color=DATASET_COLOR["plantvillage"], label="PlantVillage")
        if len(pd_rgb[i]):
            axes[i].hist(pd_rgb[i], bins=40, range=(0, 255), alpha=0.5, density=True,
                         color=DATASET_COLOR["plantdoc"], label="PlantDoc")
        axes[i].set_title(chan_name); axes[i].set_xlabel("pixel value (0-255)")
    axes[0].set_ylabel("density"); axes[0].legend()
    fig.suptitle("Channel value distribution (sampled)", y=1.03, fontsize=13, fontweight="bold")
    return _savefig(fig, out_dir, "11_channel_value_distribution.png")


# ---------------------------------------------------------------------------
# 12. file size distribution
# ---------------------------------------------------------------------------
def plot_file_size_distribution(pv_df: pd.DataFrame, pd_df: pd.DataFrame, out_dir: str | Path = EDA_OUT_DIR,
                                 n_sample: int = 1000, seed: int = 42) -> Path:
    """Histogram of on-disk file size (KB), both datasets overlaid --
    cheap signal for unusually large/corrupt/odd files, and for estimating
    total dataset size / Kaggle I/O cost."""
    import os
    _section("12. File size distribution (sampled)")

    def sample_kb(df, n):
        sizes = []
        for fp in _sample_filepaths(df, n, seed):
            try:
                sizes.append(os.path.getsize(fp) / 1024)
            except OSError:
                pass
        return np.array(sizes)

    pv_kb = sample_kb(pv_df, n_sample)
    pd_kb = sample_kb(pd_df, n_sample)
    for name, kb in [("PlantVillage", pv_kb), ("PlantDoc", pd_kb)]:
        if len(kb):
            print(f"{name}: {len(kb)} files sampled | median {np.median(kb):.1f} KB | "
                  f"min {kb.min():.1f} KB | max {kb.max():.1f} KB")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(pv_kb):
        ax.hist(pv_kb, bins=40, alpha=0.5, color=DATASET_COLOR["plantvillage"], label="PlantVillage")
    if len(pd_kb):
        ax.hist(pd_kb, bins=40, alpha=0.5, color=DATASET_COLOR["plantdoc"], label="PlantDoc")
    ax.set_xlabel("file size (KB)"); ax.set_ylabel("count")
    ax.set_title("File size distribution (sampled)")
    ax.legend()
    return _savefig(fig, out_dir, "12_file_size_distribution.png")


# ---------------------------------------------------------------------------
# 13. sample image grid (one dataset)
# ---------------------------------------------------------------------------
def plot_sample_grid(df: pd.DataFrame, dataset_name: str, out_dir: str | Path = EDA_OUT_DIR,
                      classes: list[str] | None = None, n_classes: int = 6, n_per_class: int = 4,
                      seed: int = 42) -> Path:
    """Grid of real sample thumbnails, rows=classes x cols=n_per_class --
    the "actually look at the images" sanity check. Defaults to the
    `n_classes` largest joint classes if `classes` isn't given."""
    _section(f"13. Sample image grid -- {DATASET_LABEL[dataset_name]}")
    rng = random.Random(seed)
    if classes is None:
        classes = df["joint_key"].value_counts().head(n_classes).index.tolist()
    print(f"Classes shown: {classes}")

    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2.1 * n_per_class, 2.3 * len(classes)))
    # np.atleast_2d would mis-shape a single-column grid (len(classes)>1, n_per_class==1):
    # matplotlib squeezes that case to a 1D array of length len(classes), and atleast_2d
    # reshapes 1D arrays to (1, N) rather than (N, 1), silently swapping rows for columns.
    # Reshaping against the known, requested (nrows, ncols) is correct in every case.
    axes = np.asarray(axes).reshape(len(classes), n_per_class)
    for r, cls in enumerate(classes):
        sub = df[df["joint_key"] == cls]
        picks = sub["filepath"].tolist()
        rng.shuffle(picks)
        for c in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                try:
                    with Image.open(picks[c]) as im:
                        ax.imshow(im.convert("RGB"))
                except (UnidentifiedImageError, OSError):
                    ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
            if c == 0:
                ax.text(-0.15, 0.5, cls, transform=ax.transAxes, ha="right", va="center",
                        fontsize=8, rotation=0)
    fig.suptitle(f"Sample images -- {DATASET_LABEL[dataset_name]}", y=1.0, fontsize=13, fontweight="bold")
    return _savefig(fig, out_dir, f"13_sample_grid_{dataset_name}.png")


# ---------------------------------------------------------------------------
# 14. domain comparison grid (PlantVillage vs PlantDoc, matched classes)
# ---------------------------------------------------------------------------
def plot_domain_comparison_grid(pv_df: pd.DataFrame, pd_df: pd.DataFrame, taxonomy, out_dir: str | Path = EDA_OUT_DIR,
                                 joint_classes: list[str] | None = None, n_classes: int = 4,
                                 n_per_class: int = 3, seed: int = 42) -> Path:
    """For each of a few MATCHED joint classes (present in both datasets),
    shows PlantVillage samples next to PlantDoc samples -- the single plot
    that makes the controlled-vs-field domain gap this project's regime 2/3
    exist for actually visible, rather than just a number in a report."""
    _section("14. Domain comparison -- PlantVillage vs PlantDoc (matched classes)")
    rng = random.Random(seed)
    matched = taxonomy.canonical["joint_classes_matched_both"]
    if joint_classes is None:
        # prefer matched classes with decent support in both, largest PlantDoc count first
        pd_counts = pd_df["joint_key"].value_counts()
        candidates = [c for c in matched if c in pd_counts.index]
        joint_classes = sorted(candidates, key=lambda c: -pd_counts.get(c, 0))[:n_classes]
    print(f"Classes shown (present in both datasets): {joint_classes}")

    total_cols = 2 * n_per_class
    fig, axes = plt.subplots(len(joint_classes), total_cols,
                              figsize=(1.9 * total_cols, 2.1 * len(joint_classes)))
    # see plot_sample_grid's comment above -- reshape against known dims, not atleast_2d.
    axes = np.asarray(axes).reshape(len(joint_classes), total_cols)
    for r, cls in enumerate(joint_classes):
        pv_picks = pv_df.loc[pv_df["joint_key"] == cls, "filepath"].tolist()
        pd_picks = pd_df.loc[pd_df["joint_key"] == cls, "filepath"].tolist()
        rng.shuffle(pv_picks); rng.shuffle(pd_picks)
        for c in range(n_per_class):
            ax = axes[r, c]; ax.axis("off")
            if c < len(pv_picks):
                with Image.open(pv_picks[c]) as im:
                    ax.imshow(im.convert("RGB"))
            if c == 0:
                ax.set_title("PlantVillage", fontsize=9, loc="left", color=DATASET_COLOR["plantvillage"])
        for c in range(n_per_class):
            ax = axes[r, n_per_class + c]; ax.axis("off")
            if c < len(pd_picks):
                with Image.open(pd_picks[c]) as im:
                    ax.imshow(im.convert("RGB"))
            if c == 0:
                ax.set_title("PlantDoc", fontsize=9, loc="left", color=DATASET_COLOR["plantdoc"])
        axes[r, 0].text(-0.25, 0.5, cls, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Domain comparison: same class, PlantVillage (controlled) vs PlantDoc (field)",
                 y=1.0, fontsize=12, fontweight="bold")
    return _savefig(fig, out_dir, "14_domain_comparison_grid.png")


# ---------------------------------------------------------------------------
# 15. data quality summary
# ---------------------------------------------------------------------------
def plot_data_quality_summary(stats: dict, out_dir: str | Path = EDA_OUT_DIR) -> Path:
    """Single annotated bar panel: corrupt-file counts (both datasets) and
    PlantDoc's exact-duplicate count found in its train folder -- the
    "is there anything actually wrong with the files" summary."""
    _section("15. Data quality summary")
    pv_corrupt = len(stats["plantvillage"].get("corrupt_files", []))
    pd_corrupt = len(stats["plantdoc"].get("corrupt_files", []))
    print(f"Corrupt files found -- PlantVillage: {pv_corrupt} | PlantDoc: {pd_corrupt}")
    if pv_corrupt:
        print("  PlantVillage corrupt:", stats["plantvillage"]["corrupt_files"])
    if pd_corrupt:
        print("  PlantDoc corrupt:", stats["plantdoc"]["corrupt_files"])

    labels = ["PlantVillage\ncorrupt files", "PlantDoc\ncorrupt files"]
    vals = [pv_corrupt, pd_corrupt]
    colors = [STATUS_GOOD if v == 0 else STATUS_CRITICAL for v in vals]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("count"); ax.set_title("Data quality: corrupt files")
    ax.set_ylim(0, max(1, max(vals) * 1.3))
    return _savefig(fig, out_dir, "15_data_quality_summary.png")


# ---------------------------------------------------------------------------
# run everything
# ---------------------------------------------------------------------------
def run_full_eda(manifests_dir: str | Path = "data_manifests", configs_dir: str | Path = "configs",
                  out_dir: str | Path = EDA_OUT_DIR, seed: int = 42) -> list[Path]:
    """Runs every plot function above in order, in one call. Returns the
    list of saved file paths. Prefer calling the individual functions in
    separate Kaggle cells if you want to inspect each plot as it's made;
    this is for a single "regenerate everything" run."""
    pv_df, pd_df, stats, taxonomy, label_encoders, resolved_cfg = load_eda_inputs(manifests_dir, configs_dir)

    saved = [
        plot_dataset_split_overview(pv_df, pd_df, out_dir),
        plot_joint_class_counts(pv_df, "plantvillage", taxonomy, out_dir),
        plot_joint_class_counts(pd_df, "plantdoc", taxonomy, out_dir),
        plot_crop_counts(pv_df, pd_df, out_dir),
        plot_disease_counts(pv_df, "plantvillage", out_dir),
        plot_disease_counts(pd_df, "plantdoc", out_dir),
        plot_healthy_vs_diseased(pv_df, pd_df, out_dir),
        plot_class_imbalance(pv_df, "plantvillage", out_dir),
        plot_class_imbalance(pd_df, "plantdoc", out_dir),
        plot_crop_coverage_heatmap(stats, out_dir),
        plot_joint_class_overlap(taxonomy, out_dir),
        plot_zero_support_classes(stats, out_dir),
        plot_image_size_distribution(pv_df, pd_df, out_dir, seed=seed),
        plot_channel_value_distribution(pv_df, pd_df, out_dir, seed=seed),
        plot_file_size_distribution(pv_df, pd_df, out_dir, seed=seed),
        plot_sample_grid(pv_df, "plantvillage", out_dir, seed=seed),
        plot_sample_grid(pd_df, "plantdoc", out_dir, seed=seed),
        plot_domain_comparison_grid(pv_df, pd_df, taxonomy, out_dir, seed=seed),
        plot_data_quality_summary(stats, out_dir),
    ]
    _section(f"EDA complete -- {len(saved)} figures saved under {out_dir}/")
    return saved
