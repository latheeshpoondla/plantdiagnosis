# EDA findings that actually change modelling decisions

Written after the first real (non-fixture) `run_data_pipeline.py` run on Kaggle
and a full `run_full_eda()` pass. This is deliberately *not* a recap of every
plot -- it's the subset of findings that change a training/eval decision for
the next step (model code, `src/models/`), tied back to the 4 label
formulations x 3 regimes x 2 backbones grid from the 2026-09-19 discussion
(`00_conversation_log.md`).

## 1. Class imbalance -- different severity per dataset, different fix

PlantVillage train (38 joint classes): max 4,073 (`orange__huanglongbing`),
min 108 (`potato__healthy`), **37.7x** ratio, median 801. No class under 100
images. -> inverse-frequency (or sqrt-inverse-frequency, gentler) class
weights in the loss are enough; no oversampling needed, the floor is high
enough that a pretrained backbone won't starve on any class.

PlantDoc train (27 joint classes): max 158 (`corn__northern_leaf_blight`),
min 39 (`tomato__mosaic_virus`), only **4.1x** ratio, but median is just 72
and **24/27 classes (89%) have under 100 train images**, 5 have under 50.
This is a low-data fine-tuning regime, not an imbalance problem -> the
intervention that matters here is strong augmentation + a frozen/
slowly-unfrozen pretrained backbone, with class weights as a secondary
measure, not the primary one.

## 2. PlantDoc test set per-class support is small enough to change how results get reported

27 test classes, median 9 images/class, min 4 (`corn__gray_leaf_spot`), max
12. **17/27 classes have under 10 test images.** One misclassified image
swings a class's reported accuracy by ~10 points.

-> Any PlantDoc-test number (regime 2 or regime 3) needs to be reported with
a bootstrap or Wilson CI, not a bare point estimate, and needs multiple
seeds (the 2026-09-19 plan already called for 3 seeds + paired significance
testing -- this is the concrete reason why, not just a generic best
practice). `corn__gray_leaf_spot` at n=4 is too small to support any
per-class claim at all; flag it as "insufficient support" in eval output
rather than letting it silently average into macro metrics.

## 3. Only 27 of 38 joint classes exist in both datasets -- this decides what "fair comparison" means

`joint_classes_matched_both` = 27. `joint_classes_plantvillage_only` = 11
(`apple__apple_black_rot`, `cherry__powdery_mildew`, `corn__healthy`,
`grape__esca_black_measles`, `grape__leaf_blight_isariopsis`,
`orange__huanglongbing`, `peach__bacterial_spot`, `potato__healthy`,
`strawberry__leaf_scorch`, `tomato__spider_mites`, `tomato__target_spot`).
PlantDoc has zero PlantVillage-only classes beyond this.

-> For any PVD-pretrain -> PlantDoc regime-3 comparison, decide up front
whether the pretrained head's 11 extra classes are masked out at
PlantDoc-eval time or left active as plausible distractors, and say which
in the eval code/report -- don't let it be an implicit, undocumented
choice. For regime-1-vs-regime-3 comparisons specifically, restrict to the
27 matched classes so the two regimes are being scored on the same label
space.

## 4. Ten of fourteen crops have a structural healthy/diseased blind spot

From `taxonomy_crop_healthy_diseased_coverage`:

| crop | gap |
|---|---|
| orange | absent from PlantDoc entirely; PlantVillage itself has no healthy-orange class |
| corn, potato | no healthy examples in PlantDoc (diseased-only there) |
| cherry, peach, strawberry | no diseased examples in PlantDoc (healthy-only there) |
| blueberry, raspberry, soybean | no diseased class in PlantVillage at all |
| squash | no healthy class in PlantVillage at all |

A single joint `crop__disease` classifier absorbs this fine (those crops
just have fewer joint classes). It matters for any **factorized** head (a
crop-agnostic or per-crop healthy-vs-diseased signal, relevant to the
multitask/shared-encoder formulation): for corn/potato that head will never
see a genuine PlantDoc-healthy example at eval time, and the reverse for
cherry/peach/strawberry.

-> Carry the zero-fill-don't-silently-omit discipline (already built into
`stats.class_counts()`) into the eval code too: a crop's healthy-recall on
PlantDoc should come out as "no support" rather than 0% or NaN silently
averaged into a macro score. Also worth deciding now whether `orange` gets
excluded from any healthy-vs-diseased head entirely, since the model can
structurally only ever call it diseased -- leaving it in risks it being
memorized as a free shortcut class rather than evidence the head is working.

## 5. The domain gap is real and measurable, which justifies actually running regime 1's zero-shot PlantDoc eval rather than skipping to regime 3

| | PlantVillage | PlantDoc |
|---|---|---|
| image size | fixed 256x256 | 115x150 to 6000x4000 (mean ~1109x909) |
| channel mean (RGB) | 0.469 / 0.492 / 0.414 | 0.471 / 0.531 / 0.370 |
| channel std (RGB) | 0.199 / 0.174 / 0.217 | 0.269 / 0.255 / 0.289 |
| healthy fraction | 27.8% | 33.1% |

PlantDoc's per-channel std is 25-45% higher than PlantVillage's in every
channel, and its size distribution is essentially unbounded compared to
PlantVillage's exact uniformity -- consistent with PlantVillage's clean,
single-leaf, studio-lit photos vs PlantDoc's real, cluttered, variably-lit
field photos (this is the known reason PlantVillage-only models tend to
overfit to background/framing rather than disease features).

-> Don't normalize with ImageNet stats by default if PlantDoc performance
is the actual goal -- these per-dataset values are already computed and
sitting in `dataset_stats.json`; use them (or a combined mean/std) instead.
Apply real augmentation (random-resized-crop, color jitter, possibly
cutout/random-erasing) even during **PVD-only** training, not just during
PlantDoc fine-tuning, if generalizing to PlantDoc is the point -- otherwise
regime 1's zero-shot-on-PlantDoc number is close to guaranteed to crater,
and that's worth measuring as the actual first experiment (confirms the
staged/gated plan's first step: pilot one backbone, measure whether
PVD-pretraining even helps, before expanding).

## 6. What's *not* a problem, so no special-casing needed

Zero corrupt files in either dataset (300-sample PlantVillage check, 300
in this run's PlantDoc check). Healthy/diseased overall ratio is similar
across datasets (28/72 vs 33/67) -- no extra rebalancing beyond the
class-weighting in point 1. The 2 exact-duplicate PlantDoc train images the
hash-dedupe guard caught are already removed from the resolved splits, so
no further leakage cleanup is needed there.
