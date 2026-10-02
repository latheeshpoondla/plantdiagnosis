"""Experiment directory naming + regime-1 checkpoint auto-discovery (the
mechanism behind "pretrained_source=auto automatically loads regime 1's
weights for regime 3" -- see assets/docs/05_modelling_decisions.md).
"""
from __future__ import annotations

import json
from pathlib import Path

REGIME_NAMES = {1: "regime1_plantvillage", 2: "regime2_plantdoc", 3: "regime3_pvd_pretrain_plantdoc_finetune"}
REGIME_DATASET = {1: "plantvillage", 2: "plantdoc", 3: "plantdoc"}


def experiment_dir(runs_dir: str | Path, regime: int, backbone: str, label_mode: str, seed: int) -> Path:
    return Path(runs_dir) / REGIME_NAMES[regime] / backbone / label_mode / f"seed_{seed}"


def find_regime1_checkpoint(runs_dir: str | Path, backbone: str, label_mode: str) -> Path | None:
    """Looks for a completed regime-1 run (same backbone+label_mode, any
    seed) under runs_dir and returns its best checkpoint path, or None if
    none exists yet. Regime 1 is policy-run with a single seed (42), but if
    more than one seed directory exists, picks the one with the best
    recorded validation metric rather than guessing -- never silently picks
    an arbitrary one.
    """
    base = Path(runs_dir) / REGIME_NAMES[1] / backbone / label_mode
    if not base.is_dir():
        return None

    candidates = []
    for seed_dir in sorted(base.glob("seed_*")):
        ckpt = seed_dir / "checkpoints" / "best.pt"
        summary_path = seed_dir / "training_summary.json"
        if ckpt.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            candidates.append((summary.get("best_val_metric", float("-inf")), ckpt))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_pretrained_source(pretrained_source: str, regime: int, runs_dir, backbone: str, label_mode: str):
    """Returns ('imagenet', None) or ('pvd_regime1', <checkpoint Path>) or
    ('none', None). Raises a clear, actionable error if pretrained_source
    resolves to pvd_regime1 but no regime-1 checkpoint exists yet -- never
    silently falls back to ImageNet weights, since that would make a
    regime-3 run look like it ran but secretly wasn't the experiment asked
    for.
    """
    if pretrained_source == "auto":
        pretrained_source = "pvd_regime1" if regime == 3 else "imagenet"

    if pretrained_source == "imagenet":
        return "imagenet", None
    if pretrained_source == "none":
        return "none", None
    if pretrained_source == "pvd_regime1":
        ckpt = find_regime1_checkpoint(runs_dir, backbone, label_mode)
        if ckpt is None:
            raise FileNotFoundError(
                f"--pretrained_source=pvd_regime1 (or --regime 3's default) needs a completed "
                f"regime-1 run for backbone={backbone!r}, label_mode={label_mode!r} under "
                f"{Path(runs_dir) / REGIME_NAMES[1] / backbone / label_mode}, but none was found. "
                f"Run regime 1 first: "
                f"python scripts/train.py --backbone {backbone} --regime 1 --label_mode {label_mode}"
            )
        return "pvd_regime1", ckpt

    raise ValueError(f"Unknown pretrained_source {pretrained_source!r}. Must be 'auto', 'imagenet', 'pvd_regime1', or 'none'.")
