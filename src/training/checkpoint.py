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
    """For regime 3: load ONLY the `backbone.*` weights from a regime-1
    checkpoint -- `crop_head`/`disease_head` are filtered out before
    `load_state_dict` ever sees them, so the heads always start from this
    model's own fresh random init, never from regime-1's learned head
    weights, regardless of whether the shapes happen to match. Deliberate,
    for two reasons (see assets/docs/05_modelling_decisions.md):

    1. Transfer-learning rationale: the backbone's learned visual features
       (edges, textures, disease patterns) are the generically reusable
       part; the heads are the final decision boundary fit to PlantVillage's
       own feature distribution. Given the measured PlantVillage/PlantDoc
       domain gap, carrying that boundary over could bias the fine-tune
       rather than help it -- a fresh head learns its boundary purely from
       PlantDoc's own labels from epoch 0, on top of the transferred
       backbone features.
    2. Forward-compatibility: multitask's crop/disease label spaces happen
       to be identical across regimes, so a (hypothetical) whole-model load
       would also work *today* -- but the joint-label formulation planned
       for later does NOT share head shapes between regimes (PlantVillage's
       38 joint classes vs PlantDoc's 27 matched), so a generic whole-model
       load would crash or silently misload there. Filtering to
       `backbone.*` unconditionally means this function stays correct
       without changes once that formulation exists.
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
