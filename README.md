# Plant Diagnosis

Crop + disease classification from leaf images, trained on **PlantVillage**
(controlled) and **PlantDoc** (real-world field photos), evaluated across
multiple label formulations, learning regimes and backbones (ConvNeXt-Tiny,
Swin-Tiny). See `assets/docs/` for the full write-up; this file is just the
"how do I run it" quickstart.

The project intentionally does the **data stage first, in full, before any
modelling**: every split, label mapping, and file location a model will ever
need is produced once by `scripts/run_data_pipeline.py` and written to
`data_manifests/`. Model code should never touch `/kaggle/input` directly --
it reads the manifests.

## Quickstart (on Kaggle, with the PlantVillage + PlantDoc datasets attached)

```bash
python scripts/run_data_pipeline.py
```

This resolves both datasets under `/kaggle/input` (auto-discovers the real
class-folder path even if Kaggle nests it oddly), scans every image, assigns
train/val/test splits, computes dataset statistics, and writes everything
below to `data_manifests/`:

| File | Contents |
|---|---|
| `plantvillage_manifest.csv` | one row per PlantVillage image: filepath, crop, disease, joint label, split |
| `plantdoc_manifest.csv` | same, for PlantDoc |
| `label_encoders.json` | crop / disease / joint-class name <-> id, shared by every model |
| `dataset_stats.json` | every count/measurement the run produced (class counts, image sizes, corrupt files, sampled channel mean/std) |
| `data_config_resolved.json` | the exact paths and config values that run actually used |
| `run_data_pipeline.summary.json` / `.log.jsonl` | timestamped run log |

See `assets/docs/03_data_pipeline_guide.md` for the full contract (what a
model is guaranteed about these files) and `assets/docs/01_data_analysis.md`
/ `02_taxonomy_mapping.md` for how the datasets and label taxonomy were
analysed.

## EDA

Once `data_manifests/` exists, `src/eda.py` generates every plot used to
sanity-check the data -- split balance, class counts/imbalance, crop and
disease coverage, healthy/diseased ratios, image size/channel/file-size
distributions, and real sample-image grids (including a PlantVillage-vs-
PlantDoc domain-gap comparison on matched classes). It's one function per
plot, each loading nothing on its own and printing a short finding before
saving a numbered PNG to `eda_outputs/` -- written for one call per Kaggle
cell:

```python
from src.eda import *
pv_df, pd_df, stats, taxonomy, label_encoders, resolved_cfg = load_eda_inputs()

plot_dataset_split_overview(pv_df, pd_df)
plot_joint_class_counts(pv_df, "plantvillage", taxonomy)
# ...or just:
run_full_eda()   # every plot, in order, in one call
```

## Layout

```
configs/            paths.json, data_config.json, taxonomy/*.json  (all data-stage settings + the label taxonomy)
data_meta/           small, versioned, hand-verified metadata (tracked in git) -- e.g. the
                      leaf-grouped PlantVillage split, copied from the HuggingFace mirror
src/data/             the data pipeline: paths.py (dataset auto-discovery), taxonomy.py,
                      scan.py, split.py, dataset.py (torch Dataset), transforms.py, stats.py
src/utils/            io/seed/logging helpers shared by the data stage and (later) training
src/eda.py             EDA plot functions, one per plot -- reads data_manifests/, writes eda_outputs/
scripts/               run_data_pipeline.py (the entrypoint), make_synthetic_fixture.py (dev/test only)
data_manifests/       OUTPUT of the pipeline -- gitignored, regenerated every session
eda_outputs/           OUTPUT of src/eda.py -- gitignored, regenerated every session
data/                 raw dataset downloads/clones -- gitignored, never committed
assets/docs/           the running written record of this project's data/modelling decisions
```

## Requirements

```bash
pip install -r requirements.txt
```

(Kaggle notebooks already have pandas/numpy/Pillow/torch/torchvision preinstalled.)
