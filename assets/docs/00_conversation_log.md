# Conversation log

Running record of decisions made in conversation with Claude, so context
survives independent of any chat history. Newest entries at the bottom.
This file is meant to be appended to, not rewritten, as the project goes on.

---

## 2026-09-19 -- initial experiment-design discussion

Reviewed `assets/PlantDiagnosis_Plan.md` (flagged by you as having a wrong
title and some wrong info -- kept only as a rough statement of intent, not
followed literally) plus PlantVillage/PlantDoc's public repos.

Scope agreed: crop classifier and disease classifier built separately, and
also jointly two ways (joint labels; shared encoder with separate crop/
disease heads) -- 4 formulations total. Each run under 3 learning regimes
(PlantVillage train+test; PlantDoc train+test; PlantVillage-pretrained then
PlantDoc train+test) and 2 backbones (ConvNeXt-Tiny, Swin-Tiny) -- a 4x3x2 =
24-cell grid. Leaf-separation (segmentation/detection) and the VLM
explanation layer are explicitly deferred to a later discussion.

Key finding from that grid analysis: only 8 of the 24 cells are "expensive"
(PlantVillage-scale) trainings -- regime 3 reuses regime 1's checkpoint, so
the other 16 cells are cheap PlantDoc-scale fine-tunes, and regime-1 models
can be evaluated zero-shot on PlantDoc test for free. PlantDoc's test set is
small (order of a few hundred images), so seed noise (~+-5pt 95% CI at that
size) needs 3 seeds and paired significance testing, not a finer grid, to
tell real gaps from noise. Recommended a staged, gated approach (pilot on
one backbone -> decide if PVD-pretraining even helps -> expand formulations
on the winning backbone -> confirm on the second backbone) rather than
running the full grid blind.

Also corrected the plan's claim that PlantDoc has no leaf bounding boxes --
it does, in a separate Object-Detection release of the same dataset (see
`01_data_analysis.md`).

---

## 2026-09-28 -- data stage: full pipeline built

Explicit instruction from you: focus on data / preprocessing / splits only,
build it completely before any model code, make it Kaggle-session-ready and
plug-and-play for whichever formulation/regime/backbone comes next.

Found real local data while investigating: `data/pvd/` is a complete working
clone of the HuggingFace `mohanty/PlantVillage` repo, with genuine
leaf-grouped official splits (`leaf_grouping/leaf-map.json`,
`splits/color_{train,test}.txt`) that the plain Kaggle folder-of-images
mirror doesn't ship. Verified this HF copy is the same underlying image set
as the Kaggle `abdallahalidev/plantvillage-dataset` mirror (9 spot-checked
per-class counts matched exactly), so its split was reused rather than
inventing a new one from scratch -- see `02_taxonomy_mapping.md` for the
full method. Also found `data/PlantDoc-Dataset/` and `data/plantdoc_src/` are
both dead, empty, interrupted git clones of the same repo (0 objects each) --
flagged for you to delete, not touched otherwise.

Built:
- `configs/taxonomy/{plantvillage,plantdoc}_classes.json` +
  `canonical_taxonomy.json` -- the full crop/disease label mapping between
  the two datasets, with two mapping assumptions explicitly flagged rather
  than silently merged (see `02_taxonomy_mapping.md`).
- `configs/paths.json` + `configs/data_config.json` -- every data-stage knob
  in one place (Kaggle dataset slugs, split fractions, seed, image
  extensions, stats sampling sizes).
- `data_meta/plantvillage/` -- the leaf-map, official HF splits, and the
  precomputed leaf-grouped train/val/test resolution (39,161 / 4,435 /
  10,709 rows), committed to git since it's small (~6MB) and expensive to
  redo (unlike the raw images, which stay gitignored).
- `src/data/` -- `paths.py` (Kaggle-layout auto-discovery, handles the known
  `color/color` double-nesting quirk), `taxonomy.py`, `scan.py`, `split.py`,
  `dataset.py` (torch Dataset, works for all 4 label formulations via a
  `label_mode` switch), `transforms.py`, `stats.py`.
- `scripts/run_data_pipeline.py` -- single entrypoint; writes manifests,
  label encoders, and a stats/resolved-config JSON that captures every
  number the run produced.
- `scripts/make_synthetic_fixture.py` -- dev-only tiny fixture generator,
  used to test the pipeline end-to-end without the real multi-GB datasets.

Validated (see `03_data_pipeline_guide.md` for exactly what): path
auto-discovery against a synthetic Kaggle-nesting fixture, the PlantVillage
split-matching logic against real filenames from `data_meta/`, the PlantDoc
duplicate-leakage guard with a dedicated unit test, and a full synthetic
end-to-end pipeline run. Not yet run against the real full-size datasets --
that requires an actual Kaggle GPU session, and should be treated as the
first real validation step for this code.

Not yet done / next: run `scripts/run_data_pipeline.py` for real on Kaggle
and sanity check `dataset_stats.json` against `01_data_analysis.md`'s
numbers; only then start on model code (`src/models/`), keeping the staged
experiment plan from 2026-09-19 as the run order.

---

## 2026-09-28 (cont.) -- real-data validation + a real perf bug caught

Ran the new code against the real local PlantVillage copy (`data/pvd/`,
54,305 actual images) rather than only the synthetic fixture. Caught a real
bug in the process: `list_images()` was calling `Path.is_file()` per
candidate file, which on this machine's mounted filesystem cost ~7ms/file --
turning a should-be-1-second scan into several minutes (first two attempts
timed out at 90s/150s before this was root-caused). Fixed by filtering on
file suffix only. Also hit a stale-`__pycache__`-on-a-network-mount trap
while diagnosing it: the fixed source file was on disk correctly but Python
kept running the old bytecode until `__pycache__/` was cleared -- worth
remembering if a future "I fixed it but nothing changed" moment happens
again on this machine.

After the fix: full real scan in 1.2s, PlantVillage split match rate
**100.0%** (54,305/54,305) against the precomputed leaf-grouped split, all
images confirmed uniformly 256x256, 0 corrupt files in a 300-image sample.
See `03_data_pipeline_guide.md`'s "Validation performed" section for the
full detail. PlantDoc could not be validated the same way -- no usable local
copy exists (see `01_data_analysis.md`); that will be the first real check
to do once this runs on Kaggle.

Also deleted `data/PlantDoc-Dataset/` and `data/plantdoc_src/` (the two dead,
empty, interrupted git clones identified earlier) and cleared stray
`__pycache__/` directories from the repo.

---

## 2026-09-28 (cont. 2) -- label-space vs label-support question

You asked two sharp questions: (1) are crop/disease/joint/multitask labels
using a common encoding across both datasets, and (2) is there data that's
only-healthy or only-diseased for a given crop.

Answers, verified against the actual taxonomy JSONs rather than recalled:
(1) yes, deliberately -- `Taxonomy` builds one shared crop/disease/joint id
space as the union across both datasets (14/22/38), specifically so regime 3
(PVD-pretrain -> PlantDoc-finetune) needs no label remapping between stages.
(2) yes, and it's a real pattern, not an edge case -- squash and orange have
no healthy class in either dataset; blueberry/raspberry/soybean have no
diseased class in either; and most importantly, **corn and potato have zero
healthy examples in PlantDoc** while **cherry, peach, and strawberry have
zero diseased examples in PlantDoc**, even though PlantVillage has both for
all five. Full table and the evaluation implications are now in
`02_taxonomy_mapping.md` section 6.

Turned this into a permanent, automatic part of the pipeline rather than
just an explanation: added `Taxonomy.crop_healthy_diseased_coverage()`
(taxonomy.py) and made `stats.class_counts()` zero-fill against the full
canonical label list instead of silently omitting classes with 0 examples
(a real, if subtle, bug -- a class with no data in a split simply never
appeared as a key before this fix). `dataset_stats.json` now always includes
`taxonomy_crop_healthy_diseased_coverage` and explicit
`zero_support_{joint,crop,disease}_classes` per split, every run, so this
doesn't have to be rediscovered by asking again later or by a model report
accidentally averaging in classes it was never trained or tested on.

---

## 2026-10-02 -- EDA module (`src/eda.py`)

Explicit request: a complete EDA script that runs on the output of
`scripts/run_data_pipeline.py` (never touches `/kaggle/input` itself),
saves every plot to a folder with clear filenames, and is written as one
function per plot so individual plots can be called from their own Kaggle
notebook cell, each printing a short finding alongside the figure.

Loaded the `dataviz` skill first and used its validated palette
(`references/palette.md`): fixed categorical hue order (never cycled), the
sequential blue ramp for the coverage heatmap, and status green/red
reserved for healthy/diseased only (never reused as a dataset color).
`plantvillage`/`plantdoc` and `train`/`val`/`test` each get one fixed color
used consistently across every figure in the file.

15 plot functions (19 PNGs via `run_full_eda()`, since 4 are called once
per dataset): dataset/split overview, joint class counts per dataset,
crop-level counts (PlantVillage vs PlantDoc, log scale), disease-level
counts, healthy-vs-diseased, class imbalance (with the max/min ratio
printed), the crop healthy/diseased coverage heatmap (direct visualization
of the `taxonomy_crop_healthy_diseased_coverage` stat from the previous
entry), joint-class overlap between datasets, zero-support classes per
split, image-size scatter, RGB channel-value histograms, file-size
histogram, a sample-image grid, a PlantVillage-vs-PlantDoc domain-gap grid
on matched classes, and a corrupt-file summary. `load_eda_inputs()` loads
everything once per notebook session; every plot function takes already-
loaded dataframes/stats as arguments rather than re-reading files.

Tested against a regenerated fixture pipeline run (not the synthetic
fixture from the data-stage build -- a fresh `run_data_pipeline.py` run)
before deployment: all 19 figures render as valid, non-empty PNGs. Caught
and fixed one real bug in testing: `plot_sample_grid` and
`plot_domain_comparison_grid` used `np.atleast_2d(axes)` to normalize
matplotlib's subplot-grid return value, but `atleast_2d` reshapes a 1D
array to `(1, N)` -- correct when the grid has one *row*, wrong when it has
one *column* (`nrows>1, ncols==1`), which matplotlib also squeezes to a
bare 1D array of length `nrows`. That silently swapped rows for columns and
threw an `IndexError` once more than one row was requested with
`n_per_class=1`. Fixed by reshaping against the actual requested
`(nrows, ncols)` instead of inferring shape from the squeezed array;
re-verified against single-row, single-column, and single-cell grids in
both functions.

Also added `matplotlib` to `requirements.txt`, added `/eda_outputs/` to
`.gitignore` (same reasoning as `/data_manifests/` -- regenerated output,
not committed), and documented usage in `README.md`.

---

## 2026-10-02 (cont.) -- EDA shown inline, not just saved

Changed `_savefig()` (the one helper every plot function in `src/eda.py`
routes through) to call `plt.show()` before `plt.close(fig)`, so every plot
renders inline in the Kaggle cell output -- whether called one per cell or
all at once via `run_full_eda()` -- in addition to still being saved to
`eda_outputs/`.

---

## 2026-10-03 -- real pipeline run verified + EDA read for modelling implications

You ran `scripts/run_data_pipeline.py` for real on Kaggle (first non-fixture
run) and downloaded `data_manifests/` + `eda_outputs/` back locally. Checked
both: PlantVillage scanned 54,305 images with a **100.0% match rate** against
the precomputed leaf-grouped split (identical to the local-copy validation
from 2026-09-28), zero corrupt files, uniform 256x256. PlantDoc scanned 2,552
raw images, the hash-dedupe guard caught 2 exact duplicates in train, giving
the resolved split train=2,038 / val=278 / test=236. Zero corrupt files,
genuinely variable image sizes (115x150 to 6000x4000). Zero-support pattern
in `dataset_stats.json` exactly matches the 11 PlantVillage-only classes
already documented in `02_taxonomy_mapping.md` section 6 -- nothing new or
broken, log had no warnings. This is now the trustworthy baseline to build
model code against.

Then pulled out which of the EDA's findings actually change a modelling
decision (vs. which are just descriptive) and wrote them up in
`04_eda_modelling_implications.md`: the real class-imbalance severity per
dataset (37.7x PlantVillage vs 4.1x-but-low-data PlantDoc, needing different
fixes), PlantDoc's tiny per-class test support (median 9, min 4) forcing
bootstrap CIs + multi-seed reporting rather than bare point estimates, the
27-matched/11-PlantVillage-only class split deciding what "fair regime
comparison" even means, the ten crops with a structural healthy/diseased
blind spot in at least one dataset (and what that means for a factorized
head vs a single joint classifier), and the measured domain gap (channel
std 25-45% higher in PlantDoc, unbounded vs fixed image size) as the
concrete reason to actually run regime 1's zero-shot-on-PlantDoc eval as
the first real experiment, per the 2026-09-19 staged/gated plan, rather than
skipping straight to regime 3.

---

## 2026-10-03 (cont.) -- modelling-stage code: multitask heads, both backbones, staged plan 1

Decision: implement staged plan 1 (multitask formulation: shared encoder +
crop head + disease head) first, over both backbones, regime 1 before
regime 2/3, with the seed policy 1 (heavy) / 3 (light) and a single CLI
that reaches every (backbone, regime, seed, train/test) combination with
flag changes only -- all exactly as requested. Wrote the full reasoning
(why multitask first, why these two backbones, why this regime order, and
a concrete decision against every open question `04_eda_modelling_implications.md`
raised) to `assets/docs/05_modelling_decisions.md` *before* writing the
code it governs, per your instruction.

Built: `configs/model_config.json` (every modelling hyperparameter default,
same convention as `data_config.json`); `src/models/backbones.py` +
`multitask_model.py` (ConvNeXt-Tiny / Swin-Tiny via `torchvision.models`,
no `timm` needed -- confirmed both produce 768-dim features into a 14-way
crop head + 22-way disease head); `src/training/` (`class_weights.py`,
`metrics.py` with the zero-filled-not-omitted per-class convention,
`experiment.py` for the `runs/<regime>/<backbone>/<label_mode>/seed_<seed>/`
layout and regime-1-checkpoint auto-discovery, `checkpoint.py`,
`engine.py` for the shared train/eval loop); `scripts/train.py` (the single
entrypoint -- `--backbone`, `--regime`, `--mode {train,test}`, and
`--pretrained_source {auto,imagenet,pvd_regime1,none}` are the flags that
together reach every cell); `scripts/run_multi_seed.py` (runs the policy
seed count per regime and aggregates `test_metrics.json` across seeds into
`seeds_summary.json`, mean +/- std). Wrote
`assets/docs/06_training_guide.md` as the full CLI/output-layout reference
these two scripts' docstrings point to.

Confirmed concretely in code, not just by description: per-epoch train/val
loss+accuracy land in `epoch_metrics.csv` (one row appended per epoch, so a
crashed run still keeps every completed epoch's numbers); test evaluation
always reloads `checkpoints/best.pt` (the best-`val_metric` epoch) before
scoring, logging which epoch that was, and never touches `last.pt`; regime
3 auto-loads its backbone's regime-1 checkpoint via `--pretrained_source
auto` (the default), or fails loudly with the exact rerun command if
regime 1 hasn't been run yet for that backbone.

Tested end-to-end against the small real-image fixture (`make_synthetic_fixture.py`'s
output) with real torch/torchvision installed: both backbones' forward/
backward pass; dry-run smoke test; a real training run with early stopping
(`--patience N` correctly stops N non-improving epochs after the best one);
regime 3's auto-load for both backbones, including swin_tiny correctly
loading a swin_tiny (not convnext) regime-1 checkpoint; standalone `--mode
test` cross-dataset zero-shot eval; and `run_multi_seed.py`'s full
subprocess-per-seed loop plus its mean/std aggregation, checked against
hand-computed values. One real bug caught in review (not from a test
failure): `torch.autocast(device_type="cuda", ...)` was hardcoded in
`src/training/engine.py`, which would be wrong on a CPU-only run even
though it's gated off by `use_amp`; fixed to `device_type=device.type`. One
deprecation warning fixed: `torch.cuda.amp.GradScaler` -> `torch.amp.GradScaler("cuda", ...)`.

Also fixed a real bug found in `.gitignore` while updating it for this
phase: a blanket `/assets/` entry was silently excluding every file in
`assets/docs/` (this project's own running decision record, including this
log) from version control. Removed it, and added `/runs/` (training
outputs -- same regenerated-every-run reasoning as `/data_manifests/` and
`/eda_outputs/`).

Known limitation, sandbox-only: this dev sandbox has no GPU, a ~6 GB
process memory cap, and blocks the ImageNet-weight download host. A
forward pass at the real default `--batch_size 64` on CPU exceeds that cap
before finishing one batch (confirmed via `py-spy` -- genuinely still
inside the first forward pass, not a leak or a hang). All testing above
used `--batch_size 4` and/or `--pretrained_source none` to work within the
sandbox; the real defaults were never exercised end-to-end here and should
be watched on the first real Kaggle run (which has a GPU and normal
internet access, so neither constraint should apply there).

Next: run regime 1 for both backbones on Kaggle for real, read
`test_metrics.json`, and decide -- per the 2026-09-19 staged/gated plan --
whether to proceed to regime 2/3 or revisit a decision first.

---

## 2026-10-03 (cont.) -- fixed test-time metric suite

You specified the exact metric suite every test/eval pass must report:
Cross-Entropy Loss, Top-1 Accuracy, Top-3 Accuracy, Balanced Accuracy,
Macro-F1, Confusion Matrix. Added `top_k_accuracy`, `balanced_accuracy`,
`confusion_matrix` to `src/training/metrics.py` (macro_f1/accuracy/
per_class_report already existed); `src/training/engine.py: run_epoch` now
also collects each head's top-3 predicted ids per batch when building the
final test report (not every val epoch -- early stopping doesn't need it),
and `build_test_report` assembles the full per-head suite plus
mean-of-both-heads summaries at the top level. `scripts/run_multi_seed.py`'s
`seeds_summary.json` now aggregates mean+/-std for all five metrics per
head, not just accuracy/macro-F1. Wrote the full reasoning (why each
metric, which EDA finding it answers) to `05_modelling_decisions.md`,
"Test-time metric suite", and the exact `test_metrics.json` shape to
`06_training_guide.md`.

Kept `accuracy` as an alias of the new `top1_accuracy` so nothing reading
the old key breaks. Tested: `metrics.py`'s three new functions against a
synthetic 4-class example with one zero-support class (balanced_accuracy
matches hand-computed mean-of-recalls; the confusion matrix's zero-support
row is all-zero; every row sums to its class's support); then
`scripts/train.py` and `run_multi_seed.py` run end-to-end against the
fixture, confirming top3 >= top1 for both heads, 14x14/22x22 confusion
matrices with correct row sums, and `seeds_summary.json` carrying all five
metrics' mean/std per head.
