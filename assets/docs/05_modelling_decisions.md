# Modelling-stage decisions (written before/while implementing `src/models/`, `src/training/`, `scripts/train.py`)

This picks one concrete choice for every open question `04_eda_modelling_implications.md`
raised, so the implementation isn't making silent calls. Each decision says
which EDA finding it answers and how it's exposed as a CLI flag (so it can
be revisited later without touching code).

## Scope for this pass

**Formulation:** multitask -- one shared encoder, two independent linear
heads (`crop_head`: 14-way, `disease_head`: 22-way). Chosen first (over
crop-only/disease-only/joint) because it sidesteps EDA finding #3 entirely:
`crop` and `disease` are both fully shared label spaces across PlantVillage
and PlantDoc (every crop and every disease value exists in both datasets'
taxonomy definitions -- `healthy` is itself one of the 22 disease values).
Only specific *(crop, disease)* combinations are missing in one dataset
(finding #4), which a factorized multitask head is naturally robust to:
each head is scored on its own full, shared label space, never on a
combination that might not exist. The joint/crop-only/disease-only
formulations, which DO run into the 27-vs-38-class issue, come later.

**Backbones:** ConvNeXt-Tiny and Swin-Tiny, both via `torchvision.models`
(`convnext_tiny`, `swin_t`), both ImageNet-pretrained by default -- no new
dependency on `timm`.

**Regime order:** regime 1 (PlantVillage train+test) implemented and run
first. Regime 2 (PlantDoc-only) and regime 3 (PlantVillage-pretrain ->
PlantDoc fine-tune) use the *same* code, selected by `--regime`, held back
until regime 1's results are in -- per the 2026-09-19 staged/gated plan
(pilot one cell, decide if PVD-pretraining helps, before expanding).

**Seeds:** regime 1 (PlantVillage-scale, "heavy") runs with **1 seed**
(`42`, from `configs/data_config.json`). Regime 2/3 (PlantDoc-scale,
"light") run with **3 seeds** (`42, 43, 44`) via `scripts/run_multi_seed.py`,
directly because of EDA finding #2 (PlantDoc test median 9 images/class --
single-run per-class numbers aren't trustworthy).

## Decisions answering specific EDA findings

**Class weighting (finding #1).** `CrossEntropyLoss(weight=...)` per head,
with per-class weights computed fresh from the *actual* train split in use
at run time (`src/training/class_weights.py`), not hand-typed or read from
a possibly-stale `dataset_stats.json`. Default mode is inverse-frequency
(`--class_weight_mode inv_freq`); `sqrt_inv_freq` (gentler) and `none` are
also exposed, since PlantDoc's problem is really low per-class N rather
than imbalance ratio -- worth comparing both empirically rather than
assuming inv_freq is best for the light regimes.

**Normalization stats (finding #5).** `--norm_stats auto` (default) uses
the *real* channel mean/std already computed in `dataset_stats.json` for
whichever dataset is actually being trained/evaluated on -- PlantVillage's
own stats for regime 1, PlantDoc's for regime 2/3 -- instead of defaulting
to ImageNet stats. `imagenet` is exposed as an explicit override for an
ablation, since the backbone's pretrained conv/attention stem was itself
trained under ImageNet normalization and it's a fair question whether
switching away from it helps or hurts once fine-tuning has run long enough
to adapt.

**Augmentation (finding #5).** `src/data/transforms.py` (already written
during the data stage) already applies `RandomResizedCrop` + horizontal
flip + color jitter for every `split="train"` call -- this applies to
regime 1 too, not just PlantDoc fine-tuning, which is the augmentation
decision finding #5 called for. No change needed there, just confirming it
was already built in rather than something to add now.

**27-vs-38 joint-class masking (finding #3).** Not applicable to multitask
(see Scope above) -- revisit when the joint-label formulation is built.

**Crop healthy/diseased blind spots (finding #4).** Not specially handled
in training (multitask's heads are scored on their full shared label
spaces regardless). It DOES matter for per-class test reporting: a crop's
disease-head accuracy conditioned on that one crop (e.g. "corn, healthy
vs diseased") can have zero test support in PlantDoc for some combinations
-- `src/training/metrics.py`'s per-class breakdown is zero-filled against
the full label list (same convention as `src/data/stats.py`), never
silently omitted, so a combination with no test examples shows as
`support: 0` rather than a misleadingly perfect or missing score.

**PlantDoc test set CI width (finding #2).** Not a training-time decision
-- handled by the 3-seed policy above plus `scripts/run_multi_seed.py`
reporting mean +/- std across seeds rather than a single run's number.

## Test-time metric suite

Every test/eval pass (`scripts/train.py --mode train`'s post-training test,
and standalone `--mode test`) reports a fixed set of metrics for **each
head** (crop, disease) -- not a single accuracy number, because this
project's own EDA findings (severe PlantVillage imbalance, thin PlantDoc
per-class support) make a bare top-1 accuracy misleading on its own:

- **Cross-Entropy Loss** -- the same per-example-mean loss already computed
  for training, reported at test time too (`cross_entropy_loss`, both per
  head and combined).
- **Top-1 Accuracy** (`top1_accuracy`, `accuracy` kept as an alias) --
  standard accuracy, unweighted by class.
- **Top-3 Accuracy** (`top3_accuracy`) -- correct if the true class is
  anywhere in the head's top-3 logits (`src/training/engine.py: TOP_K = 3`).
  Meaningful here specifically because disease (22-way) and even crop
  (14-way) have enough classes, and enough visually-similar diseases within
  a crop, that "did the model get it in its top few guesses" is a
  genuinely different question from top-1, not a vanity number.
- **Balanced Accuracy** (`balanced_accuracy`) -- mean per-class recall over
  classes with nonzero test support (equal to `sklearn.metrics.
  balanced_accuracy_score`, reimplemented without the dependency). This is
  the direct fix for finding #1 (PlantVillage's 37.7x imbalance): plain
  accuracy on an imbalanced test set is dominated by the majority classes,
  balanced accuracy isn't.
- **Macro-F1** (`macro_f1`) -- already in place from the first implementation
  pass, mean F1 over classes with nonzero support.
- **Confusion Matrix** (`confusion_matrix`: `{labels, matrix}`) -- full
  NxN, zero-filled against the *complete* taxonomy (same convention as
  `per_class`/finding #4's zero-support classes) -- a class with zero test
  examples still gets an all-zero row rather than being dropped from the
  matrix.

All five (barring the matrix) also get a `mean_<metric>` at the top level
of the report, averaging crop and disease. `scripts/run_multi_seed.py`'s
`seeds_summary.json` reports mean +/- std of all five across seeds, per
head -- the confusion matrix is per-seed only in `test_metrics.json`
(averaging/summing matrices across seeds isn't done; read the individual
seeds' matrices directly if needed).

Implementation: `src/training/metrics.py` (`top_k_accuracy`,
`balanced_accuracy`, `confusion_matrix`, alongside the existing `accuracy`/
`per_class_report`/`macro_f1`); `src/training/engine.py: run_epoch` now
also collects each head's top-3 predicted ids per batch when
`collect_predictions=True` (the final test pass only -- not every val
epoch, since early stopping doesn't need it), and `build_test_report` wires
all of it into the per-head report.

## Mechanics (how the CLI stays "small option changes" across every run)

- `--regime {1,2,3}` is the one flag that decides train/val/test dataset
  (PlantVillage vs PlantDoc) and, together with `--pretrained_source auto`
  (default), whether the run starts from ImageNet weights or auto-loads
  the matching regime-1 checkpoint for that backbone+label_mode. Regime 3
  is deliberately *not* a separate script -- it's `--regime 2`'s exact
  command with `--regime 3` swapped in.
- `--mode {train,test}` (default `train`) separates "train from scratch,
  validate every epoch, test the best epoch" from "just evaluate an
  existing checkpoint on a dataset/split of choice" -- the second form is
  what makes a regime-1 model's zero-shot PlantDoc evaluation (the
  2026-09-19 plan's "free" eval) a couple of flags on the same script
  rather than a separate one.
- Every run's full resolved config, per-epoch train/val losses, and final
  test metrics land under one experiment directory, keyed by
  regime/backbone/label_mode/seed -- see `03_data_pipeline_guide.md`'s
  sibling doc, `assets/docs/06_training_guide.md`, for the exact layout
  and example commands.
