# Training guide (`scripts/train.py` / `scripts/run_multi_seed.py` reference)

`05_modelling_decisions.md` is *why* each default was picked; this doc is
*how* to actually run things -- the exact flags, the output layout, and the
commands for every run in the staged plan. One script handles every
(backbone x regime x seed) combination; nothing here needs a second script
or a code change to run a different cell of the grid.

## Quickstart -- the staged plan, in order

Run these on Kaggle, with `scripts/run_data_pipeline.py` already having
produced `data_manifests/` (see `03_data_pipeline_guide.md`).

```bash
# --- Stage 1: regime 1 (PlantVillage train+test), both backbones, 1 seed each ---
python scripts/train.py --backbone convnext_tiny --regime 1
python scripts/train.py --backbone swin_tiny     --regime 1

# Look at runs/regime1_plantvillage/<backbone>/multitask/seed_42/test_metrics.json
# before deciding to move on, per the staged/gated plan.

# --- Stage 2: regime 2 (PlantDoc-only), both backbones, 3 seeds each ---
python scripts/run_multi_seed.py --backbone convnext_tiny --regime 2
python scripts/run_multi_seed.py --backbone swin_tiny     --regime 2

# --- Stage 3: regime 3 (PlantVillage-pretrain -> PlantDoc fine-tune) ---
# Auto-loads each backbone's regime-1 checkpoint from Stage 1 -- same command
# shape as regime 2, only --regime changes.
python scripts/run_multi_seed.py --backbone convnext_tiny --regime 3
python scripts/run_multi_seed.py --backbone swin_tiny     --regime 3

# --- "Free" eval: a regime-1 model, zero-shot on PlantDoc, no fine-tuning ---
python scripts/train.py --backbone convnext_tiny --regime 1 --seed 42 \
    --mode test --eval_dataset plantdoc --eval_split test
python scripts/train.py --backbone swin_tiny     --regime 1 --seed 42 \
    --mode test --eval_dataset plantdoc --eval_split test
```

Every run above is the *same* two scripts with small flag changes -- no
separate code path per backbone, regime, or seed count.

## `scripts/train.py` -- single-run training/eval

```
python scripts/train.py --backbone {convnext_tiny,swin_tiny} --regime {1,2,3} [options]
```

Required: `--backbone`, `--regime`. Everything else defaults from
`configs/model_config.json`, resolved per-regime (heavy vs light), and the
*actual* resolved value for every flag is written to that run's
`config.json` -- nothing is ever silently defaulted without a record.

| Flag | Default | Meaning |
|---|---|---|
| `--label_mode` | `multitask` | Only formulation implemented so far (crop head + disease head). |
| `--mode` | `train` | `train`: train, validate every epoch, test the best epoch. `test`: evaluate an existing checkpoint only -- no training. |
| `--seed` | regime's policy seed (`42` heavy, `42`/`43`/`44` light -- picks the first if unset) | |
| `--epochs` | 15 (heavy) / 25 (light) | |
| `--batch_size` | 64 | |
| `--lr` / `--weight_decay` | 1e-4 / 0.05 | AdamW defaults. |
| `--optimizer` | `adamw` | `adamw` or `sgd` (momentum 0.9). |
| `--lr_schedule` | `cosine` | Cosine decay with linear warmup, or `none`. |
| `--warmup_epochs` | 1 | |
| `--class_weight_mode` | `inv_freq` | `inv_freq`, `sqrt_inv_freq`, or `none` -- computed fresh from the real train split every run, never from a stale stats file. |
| `--norm_stats` | `auto` | `auto` = the real per-dataset channel mean/std from `dataset_stats.json`; `imagenet`/`plantvillage`/`plantdoc` force a specific source. |
| `--pretrained_source` | regime's default (`imagenet` for regime 1/2, `pvd_regime1` for regime 3) | `auto`/`imagenet`/`pvd_regime1`/`none`. `pvd_regime1` auto-discovers the best seed's `best.pt` under this backbone+label_mode's regime-1 run directory; raises a clear `FileNotFoundError` naming the exact command to run first if none exists yet. |
| `--val_metric` | `mean_acc` | What decides the best epoch (and so, what checkpoint `--mode test` evaluates): `mean_acc`, `loss` (minimized), `crop_acc`, `disease_acc`. |
| `--patience` | 6 | Early-stop after this many epochs with no `val_metric` improvement. `0` disables. |
| `--num_workers` | 4 | |
| `--amp` / `--no-amp` | on | Mixed precision; no-ops on CPU automatically. |
| `--image_size` | 224 (both backbones) | |
| `--dropout` | 0.2 | Applied to the shared feature vector before both heads. |
| `--loss_weight_crop` / `--loss_weight_disease` | 1.0 / 1.0 | Weights on the two heads' summed loss. |
| `--manifests_dir` / `--configs_dir` / `--runs_dir` | `data_manifests/` / `configs/` / `runs/` | |
| `--device` | `auto` | `auto`/`cpu`/`cuda`. |
| `--dry_run` | off | 2 batches, 1 epoch, no test -- smoke-tests the pipeline; not a real run. |

`--mode test` only: `--checkpoint` (defaults to this run's own `best.pt`),
`--eval_dataset` (defaults to this regime's own dataset -- set it to the
*other* dataset for a cross-dataset/zero-shot check), `--eval_split`
(default `test`).

## `scripts/run_multi_seed.py` -- the seed policy, automated

```
python scripts/run_multi_seed.py --backbone {convnext_tiny,swin_tiny} --regime {1,2,3} [--seeds 42,43,44] [any scripts/train.py flag]
```

Runs `scripts/train.py` once per seed (subprocess per seed, so a mid-run
crash on one seed doesn't lose the others' results) and aggregates the
seeds' `test_metrics.json` into `seeds_summary.json` (mean +/- std, per
head, for every metric in the fixed test-time suite -- top-1/top-3/
balanced accuracy and macro-F1; see "Test-time metric suite" below) one
directory level above the seed folders. `--seeds` defaults to `configs/model_config.json`'s policy for that
regime's weight class -- **1** for regime 1 (heavy), **3** (`42,43,44`) for
regime 2/3 (light) -- directly implementing "1 seed for heavy runs, 3 seeds
for light runs." Every flag other than `--seeds`/`--backbone`/`--regime`/
`--label_mode`/`--configs_dir`/`--runs_dir` is forwarded to `train.py`
unchanged, so it never drifts out of sync with that script's own flags.
Nothing stops running extra seeds through this for regime 1 too, if wanted.

## Output layout

Every run lands under `runs/<regime_name>/<backbone>/<label_mode>/seed_<seed>/`:

```
runs/
  regime1_plantvillage/
    convnext_tiny/multitask/seed_42/
      config.json              # every resolved hyperparameter for this exact run
      train.log.jsonl          # one JSON line per logged event (start, epoch_end, done, ...)
      epoch_metrics.csv        # per-epoch train+val loss/acc -- the thing the user asked to have stored
      checkpoints/
        best.pt                # best val_metric epoch -- this is what gets tested and what regime 3 loads
        last.pt                # most recent epoch, for resuming/inspection only
      training_summary.json    # best_epoch, best_val_metric, dry_run flag
      test_metrics.json        # test-set metrics from the BEST epoch's checkpoint (never the last epoch)
      eval_plantdoc_test.json  # only if --mode test --eval_dataset plantdoc was run from this seed dir
    swin_tiny/multitask/seed_42/   # same layout
  regime2_plantdoc/
    convnext_tiny/multitask/
      seed_42/ seed_43/ seed_44/   # same per-seed layout as above
      seeds_summary.json            # mean +/- std across the 3 seeds, written by run_multi_seed.py
  regime3_pvd_pretrain_plantdoc_finetune/
    convnext_tiny/multitask/
      seed_42/ ...                 # same -- auto-loaded its backbone's regime1 checkpoint at start
```

`epoch_metrics.csv` columns: `epoch, train_loss, train_crop_loss,
train_disease_loss, train_crop_acc, train_disease_acc, train_mean_acc,
val_loss, val_crop_loss, val_disease_loss, val_crop_acc, val_disease_acc,
val_mean_acc, lr, epoch_time_sec, is_best` -- written fresh (header + row)
at the start of training and appended to after every epoch, so a run that
dies mid-training still leaves every completed epoch's numbers on disk, not
just whatever was buffered in memory.

## Test-time metric suite

`test_metrics.json` / `eval_*.json` report a fixed set of metrics for each
head -- not a bare accuracy number, because this project's own class
imbalance (PlantVillage 37.7x) and thin per-class support (PlantDoc)
findings make one misleading on its own:

```
{
  n_examples, cross_entropy_loss,
  mean_top1_accuracy, mean_top3_accuracy, mean_balanced_accuracy, mean_macro_f1,   # crop+disease averaged
  crop: {
    cross_entropy_loss, top1_accuracy, accuracy,   # "accuracy" kept as an alias of top1_accuracy
    top3_accuracy, balanced_accuracy, macro_f1,
    confusion_matrix: {labels: [...14 crop names...], matrix: [[...14x14 ints...]]},
    per_class: {<crop name>: {support, correct, predicted_count, recall, precision, f1}, ...},
  },
  disease: {...same shape, 22 diseases...},
}
```

- **Cross-Entropy Loss** -- the training loss, reported at test time too.
- **Top-1 / Top-3 Accuracy** -- Top-3 (`src/training/engine.py: TOP_K`)
  counts a prediction correct if the true class is anywhere in that head's
  top-3 logits -- a different, less punishing question than top-1 for
  22-way disease classification.
- **Balanced Accuracy** -- mean per-class recall over classes with nonzero
  support (`sklearn.metrics.balanced_accuracy_score`'s definition,
  reimplemented without the dependency) -- the direct answer to the
  imbalance finding; plain accuracy is dominated by majority classes,
  this isn't.
- **Macro-F1** -- mean F1 over classes with nonzero support.
- **Confusion Matrix** -- full NxN, `matrix[i][j]` = count of true-class-i
  examples predicted as class j, so `matrix[i][i]` is correct count and
  `sum(matrix[i])` is that class's support.

`per_class` and `confusion_matrix` are both zero-filled against the *full*
taxonomy (14 crops / 22 diseases) -- a class with 0 test examples still
appears (an all-zero confusion-matrix row; `recall`/`precision`/`f1` as
`null`, never a misleading `0.0`). Same convention as the data stage's
`stats.class_counts()` and `04_eda_modelling_implications.md`'s finding #4.
Full reasoning: `assets/docs/05_modelling_decisions.md`, "Test-time metric
suite".

## Test-on-best-epoch, concretely

`scripts/train.py` never evaluates the last epoch's weights. After the
training loop, it reloads `checkpoints/best.pt` (the epoch `--val_metric`
picked as best) before calling `run_test`, and logs `testing_best_epoch`
with that epoch number right before doing so -- so the log itself proves
which epoch the reported test numbers came from, not just the written
checkpoint file.

## Regime 3's auto-pretrained-loading, concretely

`--pretrained_source auto` (the default for `--regime 3`) resolves to
`pvd_regime1`, which looks under
`runs/regime1_plantvillage/<this backbone>/<label_mode>/seed_*/` for a
completed run (a `checkpoints/best.pt` + `training_summary.json` pair) and
loads that checkpoint's backbone weights (`src/training/checkpoint.py:
load_backbone_weights_only` -- backbone only, heads start fresh, though in
practice the heads are the same shape across regimes for this formulation
so it ends up loading the whole model). If more than one regime-1 seed
directory exists for that backbone, it picks the one with the best recorded
`best_val_metric` rather than guessing. If none exists yet, it raises
`FileNotFoundError` naming the exact `scripts/train.py --regime 1 ...`
command to run first -- it never silently falls back to ImageNet weights,
since that would make a "regime 3" run secretly not be the experiment that
was asked for.

## Validation performed on this code

- `src/models/backbones.py` / `src/models/multitask_model.py`: both
  `convnext_tiny` and `swin_tiny` confirmed to produce `(B, 768)` backbone
  features and correct `(B, 14)` / `(B, 22)` head outputs, with a working
  backward pass, using real torch/torchvision (2.14.1 / 0.29.1, CPU-only in
  the dev sandbox).
- `src/training/class_weights.py`, `src/training/metrics.py`,
  `src/training/experiment.py`: unit tested, including the zero-support
  (`None`, not `0.0`) per-class case, the regime-1-checkpoint-not-found
  error path, and picking the best of several regime-1 seed directories.
- `scripts/train.py` run end-to-end against a small real-image fixture
  (`make_synthetic_fixture.py`'s output -- real tiny JPGs, all 38/27
  classes) covering every mode: dry-run smoke test; a real multi-epoch
  training run with early stopping (confirmed: `best_epoch` is correctly
  picked even when a later epoch is worse, and `--patience N` stops exactly
  `N` non-improving epochs after the best one); regime 3's auto-load
  (success and clear-failure paths, both backbones -- confirmed swin_tiny's
  regime-3 run correctly found and loaded a swin_tiny regime-1 checkpoint,
  never cross-loading a different backbone's weights); standalone `--mode
  test` cross-dataset eval (a regime-1 checkpoint evaluated zero-shot on
  PlantDoc test).
- `scripts/run_multi_seed.py` run end-to-end: default seed-policy resolution
  confirmed correct (regime 1 -> `[42]`, regime 2/3 -> `[42, 43, 44]`); the
  per-seed subprocess loop; and `seeds_summary.json`'s mean/std aggregation,
  checked against hand-computed values from the two seeds' real
  `test_metrics.json` outputs.
- **Test-time metric suite** (`top_k_accuracy`, `balanced_accuracy`,
  `confusion_matrix` in `src/training/metrics.py`): unit tested against a
  synthetic 4-class example with one zero-support class -- confirmed
  `balanced_accuracy` matches a hand-computed mean-of-recalls, the
  confusion matrix's zero-support row is all-zero, and every class's
  confusion-matrix row sums to exactly its `per_class` support. Then run
  end-to-end through `scripts/train.py` and `run_multi_seed.py` against the
  fixture: confirmed `top3_accuracy >= top1_accuracy` holds for both heads
  on a real (if tiny/undertrained) model, `test_metrics.json`'s confusion
  matrices are 14x14 / 22x22 with every row-sum matching `per_class`
  support, and `seeds_summary.json` carries mean/std for all five metrics
  per head.
- **Known dev-sandbox-only limitation, not a code issue:** the sandbox this
  was tested in enforces a ~6 GB process memory limit and has no GPU. A
  forward pass through either backbone at the real default `--batch_size 64`
  on CPU grows RSS past that limit before finishing one batch (confirmed via
  `py-spy`: it was genuinely still inside the first forward pass, not stuck
  or leaking). All functional tests above therefore used `--batch_size 4`
  and/or `--pretrained_source none` (ImageNet weight downloads are also
  blocked by this sandbox's network policy). Kaggle has a GPU and normal
  internet access, so neither limitation applies there -- the real defaults
  (`batch_size 64`, `pretrained_source auto` -> real ImageNet weights) were
  never exercised end-to-end precisely because this sandbox can't, and
  should be watched on the first real Kaggle run rather than assumed safe
  purely from these tests.
