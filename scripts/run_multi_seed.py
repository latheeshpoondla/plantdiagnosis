#!/usr/bin/env python3
"""Runs scripts/train.py once per seed and aggregates the test results --
the mechanism behind "1 seed for heavy runs, 3 seeds for light runs"
(assets/docs/05_modelling_decisions.md). Every flag other than --seeds is
forwarded to train.py unchanged, so this is a thin wrapper, not a second
place argument defaults can drift.

    # regime 2 (PlantDoc-only, "light"), default 3 seeds from configs/model_config.json
    python scripts/run_multi_seed.py --backbone convnext_tiny --regime 2

    # override seeds, forward any other train.py flag straight through
    python scripts/run_multi_seed.py --backbone swin_tiny --regime 3 --seeds 42,43,44 --epochs 30

A regime-1 ("heavy") run normally doesn't need this -- just call
scripts/train.py directly with its single policy seed -- but nothing stops
you from running it here too if you want extra regime-1 seeds for its own
sake.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.experiment import experiment_dir
from src.utils.io_utils import read_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["convnext_tiny", "swin_tiny"])
    p.add_argument("--regime", required=True, type=int, choices=[1, 2, 3])
    p.add_argument("--label_mode", default="multitask")
    p.add_argument("--seeds", default=None, help="Comma-separated, e.g. 42,43,44. Default: configs/model_config.json's seed policy for this regime's weight class.")
    p.add_argument("--configs_dir", default=str(REPO_ROOT / "configs"))
    p.add_argument("--runs_dir", default=str(REPO_ROOT / "runs"))
    return p.parse_known_args()  # unknown args are forwarded verbatim to train.py


def mean_std(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {"mean": mean, "std": var ** 0.5, "n": n, "values": values}


def main():
    args, passthrough = parse_args()
    model_cfg = read_json(Path(args.configs_dir) / "model_config.json")

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        weight_class = model_cfg["regimes"][str(args.regime)]["weight"]
        seeds = model_cfg["seeds"][weight_class]

    print(f"Running {len(seeds)} seed(s) for backbone={args.backbone} regime={args.regime}: {seeds}")

    for seed in seeds:
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
            "--backbone", args.backbone, "--regime", str(args.regime), "--label_mode", args.label_mode,
            "--seed", str(seed), "--configs_dir", args.configs_dir, "--runs_dir", args.runs_dir,
            *passthrough,
        ]
        print(f"\n=== seed {seed} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # aggregate test_metrics.json across all seeds
    per_seed = {}
    for seed in seeds:
        exp_dir = experiment_dir(args.runs_dir, args.regime, args.backbone, args.label_mode, seed)
        test_path = exp_dir / "test_metrics.json"
        if test_path.exists():
            per_seed[seed] = read_json(test_path)
        else:
            print(f"WARNING: no test_metrics.json for seed {seed} at {test_path} -- excluded from aggregate.")

    if per_seed:
        # The project's fixed test-time metric suite (assets/docs/05_modelling_decisions.md),
        # mean +/- std across seeds, per head -- directly the mechanism the 3-seed policy
        # exists for (PlantDoc's test set is too small per class to trust one run's number).
        summary = {"backbone": args.backbone, "regime": args.regime, "label_mode": args.label_mode, "seeds": list(per_seed)}
        for head in ("crop", "disease"):
            for metric in ("top1_accuracy", "top3_accuracy", "balanced_accuracy", "macro_f1", "cross_entropy_loss"):
                summary[f"{head}_{metric}"] = mean_std([r[head][metric] for r in per_seed.values()])
        out_path = experiment_dir(args.runs_dir, args.regime, args.backbone, args.label_mode, seeds[0]).parent / "seeds_summary.json"
        write_json(out_path, summary)
        print(f"\nAggregated {len(per_seed)}/{len(seeds)} seed(s) -> {out_path}")
        for head in ("crop", "disease"):
            t1, t3, bal, f1 = (summary[f"{head}_{m}"] for m in ("top1_accuracy", "top3_accuracy", "balanced_accuracy", "macro_f1"))
            print(f"  {head}: top1={t1['mean']:.4f}+/-{t1['std']:.4f}  top3={t3['mean']:.4f}+/-{t3['std']:.4f}  "
                  f"balanced_acc={bal['mean']:.4f}+/-{bal['std']:.4f}  macro_f1={f1['mean']:.4f}+/-{f1['std']:.4f}")
    else:
        print("No completed seeds to aggregate.")


if __name__ == "__main__":
    main()
