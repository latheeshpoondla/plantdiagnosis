# Taxonomy mapping & split methodology

How `configs/taxonomy/*.json` and the PlantVillage/PlantDoc splits were
built, so the reasoning survives independent of this conversation.

## 1. Canonical label design

Both datasets' raw folder names are kept as `raw_class` (exact string, used
only to resolve a scanned file to its label). Downstream, every image gets:

- `crop` -- one of 14 canonical crop names
- `disease` -- one of 22 canonical disease names (`healthy` is one of them)
- `joint_key` = `f"{crop}__{disease}"` -- what the "joint-label" model
  formulation predicts directly, and what the "shared encoder, two heads"
  formulation's two heads predict jointly

**Disease names are kept crop-specific unless the underlying condition is
genuinely the same.** For example `Apple___Black_rot` and `Grape___Black_rot`
are visually/textually similar labels but different pathogens
(*Botryosphaeria obtusa* vs *Guignardia bidwellii*) -- these were kept as
separate canonical diseases (`apple_black_rot`, `grape_black_rot`) rather
than merged into one `black_rot` label, because a disease-only classifier
trained on a merged label would be learning a biologically meaningless
category. Same reasoning for "rust": `cedar_apple_rust` (apple) and
`common_rust` (corn) are different fungi, kept separate.

Diseases *were* merged across crops where the condition is actually the same
labelled thing: `healthy` (by definition crop-agnostic), `bacterial_spot`
(peach/pepper_bell/tomato -- *Xanthomonas* spp.), `early_blight` and
`late_blight` (potato/tomato -- same genus/pathogen respectively),
`powdery_mildew` (cherry/squash -- same fungal group). This is a judgement
call, not a certainty; if a later error analysis shows the merged classes
are hurting the disease-only head, un-merging them is a one-line change in
`configs/taxonomy/plantvillage_classes.json` / `plantdoc_classes.json`.

## 2. PlantVillage -> PlantDoc class coverage

All 27 PlantDoc joint classes have a matching PlantVillage joint class.
PlantVillage has 11 classes with no PlantDoc equivalent at all:

`apple__apple_black_rot`, `cherry__powdery_mildew`, `corn__healthy`,
`grape__esca_black_measles`, `grape__leaf_blight_isariopsis`,
`orange__huanglongbing`, `peach__bacterial_spot`, `potato__healthy`,
`strawberry__leaf_scorch`, `tomato__spider_mites`, `tomato__target_spot`.

Two of these are worth calling out specifically: **PlantDoc has no healthy
Potato or Corn class** (every PlantVillage-only entry for those two crops is
`healthy`) -- so a model fine-tuned only on PlantDoc will never see a
healthy potato/corn leaf.

## 3. Two mapping assumptions (flagged, not silently merged)

PlantDoc's labels are casual ("Corn leaf blight") where PlantVillage's are
specific ("Northern_Leaf_Blight"). Two PlantDoc classes required picking a
specific PlantVillage disease with no way to fully verify it from the label
text alone:

| PlantDoc raw class | Assumed PlantVillage equivalent | Why |
|---|---|---|
| `Bell_pepper leaf spot` | `bacterial_spot` | Bacterial_spot is pepper_bell's *only* non-healthy disease in PlantVillage |
| `Corn leaf blight` | `northern_leaf_blight` | Northern_Leaf_Blight is corn's only PlantVillage "blight" (as opposed to rust/gray-leaf-spot) |

Both are marked `"assumption": true` in `configs/taxonomy/plantdoc_classes.json`
and listed under `canonical_taxonomy.json -> assumed_mappings`, so any code
(or person) can filter them out of a "trusted" cross-dataset comparison if
that matters later.

## 4. Full PlantDoc -> canonical mapping table

Also machine-readable at `configs/taxonomy/plantdoc_classes.json`.

| PlantDoc raw class | crop | disease | healthy? | assumption? |
|---|---|---|---|---|
| Apple Scab Leaf | apple | apple_scab |  |  |
| Apple leaf | apple | healthy | yes |  |
| Apple rust leaf | apple | cedar_apple_rust |  |  |
| Bell_pepper leaf | pepper_bell | healthy | yes |  |
| Bell_pepper leaf spot | pepper_bell | bacterial_spot |  | **yes** |
| Blueberry leaf | blueberry | healthy | yes |  |
| Cherry leaf | cherry | healthy | yes |  |
| Corn Gray leaf spot | corn | gray_leaf_spot |  |  |
| Corn leaf blight | corn | northern_leaf_blight |  | **yes** |
| Corn rust leaf | corn | common_rust |  |  |
| Peach leaf | peach | healthy | yes |  |
| Potato leaf early blight | potato | early_blight |  |  |
| Potato leaf late blight | potato | late_blight |  |  |
| Raspberry leaf | raspberry | healthy | yes |  |
| Soyabean leaf | soybean | healthy | yes |  |
| Squash Powdery mildew leaf | squash | powdery_mildew |  |  |
| Strawberry leaf | strawberry | healthy | yes |  |
| Tomato Early blight leaf | tomato | early_blight |  |  |
| Tomato Septoria leaf spot | tomato | septoria_leaf_spot |  |  |
| Tomato leaf | tomato | healthy | yes |  |
| Tomato leaf bacterial spot | tomato | bacterial_spot |  |  |
| Tomato leaf late blight | tomato | late_blight |  |  |
| Tomato leaf mosaic virus | tomato | mosaic_virus |  |  |
| Tomato leaf yellow virus | tomato | yellow_leaf_curl_virus |  |  |
| Tomato mold leaf | tomato | leaf_mold |  |  |
| grape leaf | grape | healthy | yes |  |
| grape leaf black rot | grape | grape_black_rot |  |  |

## 5. Split methodology

### PlantVillage: reuse the real leaf-grouped official split

The HuggingFace `mohanty/PlantVillage` repo (a complete copy of which is at
`data/pvd/` on this machine) ships something the raw Kaggle folder-of-images
mirror doesn't: a `leaf_grouping/leaf-map.json` (40,328 leaf identifiers ->
which photos are the same physical leaf) and pre-built, leaf-aware
`splits/color_{train,test}.txt`. This project uses that split rather than
inventing a new one, because it's a real leakage guard that's expensive to
reproduce (the leaf-id derivation depends on parsing each dataset
photographer's filename convention -- see `data/pvd/plant_village.py`,
`_generate_examples`) and free to reuse.

Concretely, run once (already done, output committed to `data_meta/`):

1. Load `splits/color_train.txt` (43,596 rows) and `splits/color_test.txt`
   (10,709 rows). **Verified**: summed per-class counts from these two files
   match the real per-class file counts on disk exactly, for all 38 classes,
   zero mismatches.
2. `test` is used as-is and locked -- never touched again.
3. For every `train` row, derive its `leaf_id` using the same normalization
   the HF loader script uses (strip `_final_masked`/`copy` suffixes, take
   the part after the last `___`, lowercase, look up in `leaf-map.json`,
   disambiguate by crop if the lookup is ambiguous). **24.1% of train rows**
   have no leaf-map entry (fall back to being their own singleton group --
   they just don't get grouped with anything, which is the safe default,
   not a leak).
4. Group train rows by `leaf_id` (16,124 groups from 43,596 rows). Zero
   groups span more than one class (sanity-checked).
5. Carve ~10% of train into `val`, allocating **whole leaf-groups** at a
   time, stratified per class (`random.seed(42)`), so no leaf's photos are
   split between train and val.
6. Result: **39,161 resolved-train / 4,435 resolved-val / 10,709 locked-test**.

The output -- `data_meta/plantvillage/resolved_splits/color_{train,val,test}.csv`
(columns: `class_name, file_name[, leaf_id]`) plus
`data_meta/plantvillage/plantvillage_split_summary.json` -- is small (~6MB
total) and **is committed to git**, unlike the raw images. At pipeline run
time (`src/data/split.py: attach_plantvillage_split`), every scanned Kaggle
file is matched into this split by `(raw_class, file_name)` -- not by full
path, since Kaggle's mount path differs from the path these CSVs were built
against. This match was spot-verified at 100% against real filenames before
being wired into the pipeline.

If a Kaggle run's PlantVillage copy ever has files these CSVs don't cover
(a different/updated dataset version), those rows are marked
`split=UNMATCHED` rather than silently dropped or guessed into a split, and
`dataset_stats.json`/the run log will show a non-zero `rows_unmatched`.

### PlantDoc: official train/test, val carved at run time

PlantDoc's official `train/` and `test/` folders are trusted as the
train/test boundary (`test` locked, never resampled). No leaf/photo-grouping
metadata is available for PlantDoc, so validation is carved from `train`
with a weaker but real guard instead: every train image is exact-content
hashed (MD5); identical files are grouped so a literal duplicate can never
end up on both sides of the train/val boundary; groups are then allocated to
val stratified by joint class, target 12% of train
(`configs/data_config.json -> plantdoc.val_fraction_of_official_train`).
This is computed fresh every pipeline run (`src/data/split.py:
make_plantdoc_split`), not precomputed, since PlantDoc is small enough that
re-hashing every run is cheap.

## 6. Shared label space, but NOT shared label support

Two related but different questions worth being precise about, since they
get confused easily:

**Is the label *space* shared across crop-only / disease-only / joint /
multitask, and across both datasets?** Yes, deliberately. `Taxonomy` (in
`src/data/taxonomy.py`) builds exactly one `crop_to_id` (14 entries), one
`disease_to_id` (22), and one `joint_to_id` (38) -- each the *union* across
both datasets' raw taxonomies, not a separate encoding per dataset. A model's
classifier head size and the meaning of each output index is therefore
identical whether it's looking at a PlantVillage batch or a PlantDoc batch.
This is what makes regime 3 (PlantVillage-pretrain -> PlantDoc-finetune)
work without any label remapping: the head learned during pretraining is
already sized and indexed correctly for the finetune stage.

**Does every class actually have training/test examples in both datasets?**
No -- and this matters a lot more than it might look like at first. Checked
directly against `configs/taxonomy/*.json` (a property of the dataset
designs themselves, not of any particular scan), via
`Taxonomy.crop_healthy_diseased_coverage()`:

| crop | PVD healthy | PVD #diseases | PD healthy | PD #diseases | issue |
|---|---|---|---|---|---|
| apple | yes | 3 | yes | 2 | — |
| blueberry | yes | 0 | yes | 0 | no diseased class exists anywhere |
| cherry | yes | 1 | yes | **0** | **no diseased example in PlantDoc** |
| corn | yes | 3 | **no** | 3 | **no healthy example in PlantDoc** |
| grape | yes | 3 | yes | 1 | — |
| orange | no | 1 | — | — | **absent from PlantDoc entirely**; no healthy class anywhere |
| peach | yes | 1 | yes | **0** | **no diseased example in PlantDoc** |
| pepper_bell | yes | 1 | yes | 1 | — |
| potato | yes | 2 | **no** | 2 | **no healthy example in PlantDoc** |
| raspberry | yes | 0 | yes | 0 | no diseased class exists anywhere |
| soybean | yes | 0 | yes | 0 | no diseased class exists anywhere |
| squash | no | 1 | no | 1 | no healthy class exists anywhere |
| strawberry | yes | 1 | yes | **0** | **no diseased example in PlantDoc** |
| tomato | yes | 9 | yes | 7 | — |

Three real patterns, not one edge case:

1. **Diseased-only crops** (no healthy class in *either* dataset): squash,
   orange (orange is also just absent from PlantDoc altogether). There is no
   way, with these two datasets, to ever teach or evaluate "what does a
   healthy squash leaf look like."
2. **Healthy-only crops** (no diseased class in *either* dataset):
   blueberry, raspberry, soybean.
3. **Crops PlantVillage covers fully but PlantDoc covers only half of** --
   this is the one that actually bites during training, not just at the
   edges of the taxonomy:
   - **Corn and potato**: PlantDoc has *zero* healthy examples. A model
     fine-tuned on PlantDoc (regimes 2 and 3) never sees a healthy
     corn/potato leaf during that stage. Regime 3 specifically risks the
     opposite failure of naive optimism: the PVD-pretrained head *did*
     learn healthy-corn/potato, but ordinary fine-tuning has no mechanism
     protecting classes the fine-tuning data never reinforces -- that
     knowledge can silently degrade (catastrophic forgetting) the longer
     fine-tuning runs, with nothing in a PlantDoc validation loss to catch
     it, since PlantDoc validation never contains that class either.
   - **Cherry, peach, strawberry**: the mirror image -- PlantDoc has *zero
     diseased* examples for these three. A PlantDoc-finetuned model will
     never be trained or tested on a diseased leaf of these crops, and any
     claim like "we detect cherry disease" would be unsupported by this
     data combination.

**What this means for evaluation, concretely**: `dataset_stats.json` (from
`scripts/run_data_pipeline.py`) now reports `class_counts_joint` /
`class_counts_crop` / `class_counts_disease` **zero-filled against the full
canonical label list**, plus an explicit `zero_support_*_classes` list per
split, plus `taxonomy_crop_healthy_diseased_coverage` (the table above,
computed fresh every run). Before this fix, a class with 0 examples in a
split simply never appeared as a dict key in the stats output -- easy to
miss. Now it's an explicit, impossible-to-miss `0`. Any macro-averaged
accuracy/F1 reported for PlantDoc should either exclude
`zero_support_joint_classes` for that split, or say plainly it's averaging
in classes the model was never shown -- silently doing neither would be
exactly the kind of overclaim the original plan document's "required
honesty" section warned against.
