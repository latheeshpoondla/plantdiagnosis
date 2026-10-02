# Data pipeline guide (the "plug and play" contract)

The point of doing the whole data stage before any model code is that a
model should never need to know anything about `/kaggle/input`, folder
layouts, filenames, or how splits were computed. It reads
`data_manifests/*.csv`/`*.json` and that's the entire contract. This doc is
that contract, written down.

## Running it

On Kaggle, with both datasets attached to the notebook:

```bash
python scripts/run_data_pipeline.py
```

Flags (all optional): `--configs_dir`, `--data_meta_dir`, `--out`,
`--skip_corrupt_check`, `--skip_channel_stats`. Defaults come from
`configs/paths.json` / `configs/data_config.json`.

If a dataset can't be found, it fails loudly with a `DatasetNotFoundError`
naming exactly which variant/split is missing and every base directory it
searched -- it does not silently proceed with partial data.

## What a model formulation reads

Every one of the plan's four label formulations (crop-only, disease-only,
joint-label, shared-encoder-two-heads) and all three learning regimes
(PVD-only, PlantDoc-only, PVD-pretrain-then-PlantDoc) read the *same* two
manifest files -- only which columns/splits they use differs:

```python
import pandas as pd
from src.data.taxonomy import load_taxonomy
from src.data.dataset import PlantLeafDataset
from src.data.transforms import build_transforms

taxonomy = load_taxonomy("configs")

pv = pd.read_csv("data_manifests/plantvillage_manifest.csv")
pd_ = pd.read_csv("data_manifests/plantdoc_manifest.csv")

# Regime 1 (PVD train+test), joint-label formulation:
train_ds = PlantLeafDataset(pv, split="train", label_mode="joint",
                             taxonomy=taxonomy, transform=build_transforms("train"))
test_ds  = PlantLeafDataset(pv, split="test", label_mode="joint",
                             taxonomy=taxonomy, transform=build_transforms("eval"))

# Regime 3 (PVD-pretrained -> PlantDoc), shared-encoder formulation:
finetune_ds = PlantLeafDataset(pd_, split="train", label_mode="multitask",
                                taxonomy=taxonomy, transform=build_transforms("train"))
```

`label_mode`:
- `"crop"` -> int label, `taxonomy.num_crops()` classes
- `"disease"` -> int label, `taxonomy.num_diseases()` classes
- `"joint"` -> int label, `taxonomy.num_joint()` classes
- `"multitask"` -> `(crop_id, disease_id)` tuple, for the shared-encoder/two-head model

`label_encoders.json` has the exact id<->name mapping used, so predictions
can always be mapped back to a human-readable crop/disease name without
re-deriving anything.

## Manifest columns

Both `plantvillage_manifest.csv` and `plantdoc_manifest.csv`:

| column | meaning |
|---|---|
| `dataset` | `"plantvillage"` or `"plantdoc"` |
| `variant` | `"color"` for PlantVillage; `"field"` for PlantDoc |
| `raw_class` | the original dataset's folder name (debugging/audit only) |
| `file_name` | image filename |
| `filepath` | full resolved path on this machine/session -- **not stable across sessions**, always re-run the pipeline rather than reusing an old manifest's filepaths |
| `crop`, `disease`, `is_healthy`, `joint_key` | canonical labels, see `02_taxonomy_mapping.md` |
| `source_split` | the split the dataset shipped with (`"train"`/`"test"`; PlantVillage rows also carry this from the HF split before `split` is attached) |
| `split` | **the one to use** -- `"train"` / `"val"` / `"test"` after this project's split logic (see `02_taxonomy_mapping.md`) |

## Cross-dataset (regime 3 / matched-class) filtering

To restrict PlantVillage to only the 27 classes PlantDoc also has (for a
fair zero-shot or fine-tuning comparison), filter on `joint_key`:

```python
matched = json.load(open("configs/taxonomy/canonical_taxonomy.json"))["joint_classes_matched_both"]
pv_matched = pv[pv.joint_key.isin(matched)]
```

## Numbers this pipeline reports (nothing is hand-typed elsewhere)

Every run writes the real, counted numbers to `data_manifests/dataset_stats.json`
and `data_manifests/data_config_resolved.json`: per-split/per-class image
counts, sampled image width/height stats, corrupt-file list (if any),
sampled per-channel RGB mean/std (to sanity-check against the ImageNet
defaults `src/data/transforms.py` uses), the exact paths that were resolved
under `/kaggle/input`, the split-assignment report (match rate for
PlantVillage, duplicate count for PlantDoc), and the seed. Any number quoted
in a future model report ("PlantDoc test set has N images") should be read
from this file, not from `01_data_analysis.md` (which records what was true
when it was written, from a possibly-different copy of the dataset) and
never retyped from memory.

## Validation performed on this code

- `src/data/paths.py`'s auto-discovery was tested against a synthetic
  fixture reproducing Kaggle's known `color/color` double-nesting quirk --
  resolves correctly, and correctly raises `DatasetNotFoundError` with a
  clear message when a dataset genuinely isn't there.
- `src/data/split.py: make_plantdoc_split`'s duplicate guard was unit
  tested: two files with byte-identical content are always assigned to the
  same split, never split across train/val.
- The full pipeline (`scan -> split -> stats -> write`) was run end-to-end
  against a synthetic fixture with real (tiny, generated) JPGs covering all
  38 + 27 classes -- completes cleanly, corrupt-check finds 0 bad files,
  stats/manifests/label-encoders all written and structurally correct.
- **`scan_plantvillage` + `attach_plantvillage_split` were run against the
  real, full-size local PlantVillage copy** (`data/pvd/data/raw/color`, the
  actual 54,305 images, not a fixture):
  - Scan found exactly 54,305 images across all 38 classes, in 1.2s.
  - Split match rate: **100.0%** (54,305 / 54,305) -- every real file matched
    a row in the precomputed `data_meta/plantvillage/resolved_splits/` CSVs
    by `(raw_class, file_name)`. Resulting split: 39,161 train / 4,435 val /
    10,709 test, exactly as computed in `02_taxonomy_mapping.md`.
  - A 150-image size sample came back **uniformly 256x256** -- PlantVillage's
    color images are pre-resized already (worth knowing: no wildly
    off-aspect source images to worry about, unlike PlantDoc's field photos).
  - A 300-image corrupt-check sample: 0 bad files.
- **PlantDoc has not been validated against real data** -- no usable local
  copy exists on this machine (see `01_data_analysis.md` -- both local git
  clones were dead). First real Kaggle run should treat PlantDoc's numbers
  in `dataset_stats.json` as the first real look at that dataset's actual
  file counts, and sanity-check them against `01_data_analysis.md`'s caveat
  about inconsistent published counts.

### A real performance bug this testing caught

`src/utils/io_utils.py: list_images` originally called `Path.is_file()` on
every candidate file to filter out directories before checking the
extension. On this project's dev machine (files reached through a mounted
remote filesystem), that one call added ~7ms per file -- for 54,305 images,
a several-minute scan instead of a ~1 second one. Fixed by filtering on file
suffix alone (skips the stat syscall entirely; a directory named like an
image file is not a realistic concern for these datasets). Kaggle's own
`/kaggle/input` is a fast local mount, so this specific slowdown may not
reproduce there -- but the fix is strictly faster either way and is already
in place.
