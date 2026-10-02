"""Checkpoint save/load. Deliberately thin -- just what scripts/train.py
needs to save best.pt / last.pt each epoch and to load a checkpoint either
to resume or to initialize regime 3 from a regime-1 run.
"""
from __future__ import annotations

from pathlib import Path


def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int | None = None, extra: dict | None = None) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu") -> dict:
    import torch

    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def load_backbone_weights_only(path: str | Path, model, map_location: str = "cpu") -> None:
    """For regime 3: load only the matching backbone weights from a
    regime-1 checkpoint, leaving crop_head/disease_head at their freshly
    initialized state. The backbone's learned features transfer; the heads
    are the same shape/label-space either way (crop/disease spaces are
    identical across regimes for the multitask formulation -- see
    assets/docs/05_modelling_decisions.md), so in practice this loads the
    whole model. Implemented as a backbone-only load anyway so it stays
    correct if a future formulation changes head shapes between regimes.
    """
    import torch

    state = torch.load(path, map_location=map_location)
    full_state = state["model_state_dict"]
    backbone_state = {k: v for k, v in full_state.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    # Anything "missing" here should only be the heads (expected -- they're
    # not part of backbone_state); any missing backbone.* key would mean a
    # real architecture mismatch and is worth surfacing loudly.
    unexpected_backbone_keys = [k for k in unexpected if k.startswith("backbone.")]
    if unexpected_backbone_keys:
        raise RuntimeError(
            f"load_backbone_weights_only found backbone.* keys in the checkpoint that don't "
            f"match this model's architecture: {unexpected_backbone_keys}. The regime-1 "
            f"checkpoint was likely trained with a different backbone/architecture version."
        )
