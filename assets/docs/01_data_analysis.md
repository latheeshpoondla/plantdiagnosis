# Data analysis: PlantVillage & PlantDoc

Written 2026-09-28. This is the deep-dive behind `configs/taxonomy/*.json`
and the split logic in `src/data/split.py`. Every number below was either
read directly off the real local data on this machine, or off the Kaggle
Data Explorer screenshots supplied for `abdallahalidev/plantvillage-dataset`
and `abdulhasibuddin/plant-doc-dataset` -- nothing here is estimated.

## What was actually on disk when this was written

Three things were found under `data/` in this repo:

1. **`data/pvd/`** -- a *complete, real* clone of the HuggingFace
   `mohanty/PlantVillage` dataset repo (loading script, `data/raw/{color,grayscale,segmented}`
   with real images, `leaf_grouping/leaf-map.json`, `splits/*.txt`). Verified
   working and used as the source for PlantVillage's official leaf-grouped
   split (see `02_taxonomy_mapping.md`).
2. **`data/PlantDoc-Dataset/`** and **`data/plantdoc_src/`** -- both are the
   *same* repo (`pratikkayal/PlantDoc-Dataset`) cloned twice, and both are
   **empty, failed clones**: `git count-objects` reports 0 objects, there is
   a `.git/config.lock` from an interrupted clone, and `git status` reports
   "No commits yet". These two folders contain nothing usable and are safe
   to delete (not done automatically -- ask if you want them removed; needs
   delete permission on this folder).
3. PlantDoc itself is **not** re-cloned by this project. Per your instruction
   not to clone large dataset repos, and since `PlantDoc-Dataset`/`plantdoc_src`
   above are dead anyway, the pipeline resolves PlantDoc from the Kaggle
   dataset (`abdulhasibuddin/plant-doc-dataset`) at run time instead, the
   same way it resolves PlantVillage.

## PlantVillage

- **38 classes**, `Crop___Disease` folder naming, confirmed byte-for-byte
  from the HuggingFace loader script (`data/pvd/plant_village.py`,
  `_CLASS_NAMES`) -- this is the authoritative list, now in
  `configs/taxonomy/plantvillage_classes.json`.
- **14 crops** (apple, blueberry, cherry, corn, grape, orange, peach,
  pepper_bell, potato, raspberry, soybean, squash, strawberry, tomato).
- **Images are pre-resized to a uniform 256x256** (measured on a 150-image
  sample of the real local copy -- min/max/mean all exactly 256x256). No
  aspect-ratio surprises to handle for this dataset's train pipeline.
- **54,305 images** total: `43,596` in the official train split +
  `10,709` in the official test split (`data/pvd/splits/color_{train,test}.txt`).
  The widely-quoted figure is "54,306"; the real, counted number here is
  54,305 -- a one-image discrepancy not worth chasing further.
- Nine spot-checked per-class counts (Apple x4, Blueberry, Cherry x2, Corn x3)
  were read directly off both the local `data/pvd/data/raw/color/*` folders
  and the Kaggle `abdallahalidev/plantvillage-dataset` screenshot -- **all
  nine matched exactly** (630, 621, 275, 1645, 1502, 1052, 854, 513, 1192,
  985). This is strong evidence the Kaggle mirror is the same underlying
  image set as the HuggingFace one, which is what makes matching by
  `(class_name, file_name)` across the two sources safe (see below).
- **Known caveat baked into the source data itself**: many images are
  multiple photos of the *same physical leaf* (different angle/lighting).
  The HuggingFace repo ships a `leaf-map.json` (40,328 leaf groups) and an
  official train/test split built to respect those groups. This project
  reuses that split rather than re-deriving grouping from scratch --
  see `02_taxonomy_mapping.md` for exactly how.
- Squash has **no healthy class** in PlantVillage (only `Powdery_mildew`).
  Corn and Potato are the reverse case in PlantDoc (see below).
- Two other raw folders exist in the same HF repo: `grayscale` and
  `segmented` (background-removed) versions of the *same* photos. Both are
  resolvable by the pipeline (`configs/paths.json ->
  plantvillage_variants_available`) but **excluded by default**
  (`plantvillage_variants_to_use: ["color"]`) -- this project's target
  input is real RGB camera images, and segmented masks would leak
  ground-truth leaf boundaries that a real deployment won't have.

## PlantDoc

- **27 classes**, confirmed from the Kaggle `abdulhasibuddin/plant-doc-dataset`
  Data Explorer screenshot (train/ and test/ each showing 27 class
  directories, with names visible: "Apple Scab Leaf", "Apple leaf",
  "Apple rust leaf", "Bell_pepper leaf", "Bell_pepper leaf spot", ... etc),
  cross-checked against the dataset paper (arXiv:1911.10317) and the
  upstream `pratikkayal/PlantDoc-Dataset` repo structure. Now in
  `configs/taxonomy/plantdoc_classes.json`.
- **13 crops**, a strict subset of PlantVillage's 14 (no orange in PlantDoc).
- Published image counts for this dataset are inconsistent across sources
  (2,598 vs 2,482 total; test set reported as 231, 236, 238, or 282 images
  depending on which mirror/blog post is cited) -- this is a known messy
  point about PlantDoc, not something this project can resolve without the
  literal files in front of it. **The pipeline does not hardcode any of
  these numbers.** `scripts/run_data_pipeline.py` counts the real files in
  whatever copy is attached to the Kaggle notebook and writes the true
  count to `data_manifests/dataset_stats.json` every run -- that number,
  not any blog post, is the one to trust.
- Dataset is real-world/field photography: cluttered backgrounds, multiple
  leaves per photo sometimes, varied lighting -- this is *why* it exists in
  this project's regime 2/3 (see the experiment design discussion).
- License: CC BY 4.0.
- The PlantDoc dataset labels *whole photos* (via the "Cropped-PlantDoc"
  classification folders) as one crop+disease class; leaf-level bounding
  boxes exist in a separate "Object-Detection" variant of the same repo
  (not used in this data stage -- relevant later for the leaf-separation
  step mentioned in the original plan, kept out of scope for now).

## Class taxonomy: PlantVillage vs PlantDoc

All 27 PlantDoc joint (crop, disease) classes have a corresponding
PlantVillage class; PlantVillage additionally has 11 classes PlantDoc has no
equivalent for at all (full list and the two label-mapping assumptions that
had to be made are in `02_taxonomy_mapping.md`). In short:

| | crops | disease labels (after merging identical conditions) | joint classes |
|---|---|---|---|
| PlantVillage | 14 | 22 | 38 |
| PlantDoc | 13 | 15 | 27 |
| Overlap | 13 | — | 27 (all of PlantDoc's) |

## Corrections to `assets/PlantDiagnosis_Plan.md`

You flagged the original plan file as containing wrong info/title; the one
concrete factual error worth calling out explicitly: **PlantDoc does have
per-leaf bounding box annotations** (in its separate Object-Detection
release), contrary to the plan's "no leaf boxes, no leaf masks" framing for
that dataset. Doesn't change anything in this data stage (we're using the
classification folders either way), but it means the later leaf-detector
step may not need a from-scratch annotation effort.
