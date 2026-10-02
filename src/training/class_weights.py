"""Per-class loss weights, computed fresh from the actual train split in
use at run time -- never read from a possibly-stale dataset_stats.json.
See assets/docs/05_modelling_decisions.md's "Class weighting" section for
why this is the decision (and why it's a CLI-exposed mode rather than one
hardcoded scheme).
"""
from __future__ import annotations

import pandas as pd


def compute_class_weights(
    train_df: pd.DataFrame,
    column: str,
    id_map: dict,
    mode: str = "inv_freq",
) -> list[float]:
    """train_df: the TRAIN split only (never val/test -- weights must come
    from what the model actually trains on). column: 'crop' or 'disease'.
    id_map: taxonomy.crop_to_id or taxonomy.disease_to_id. Returns a list of
    length len(id_map), index-aligned to id, ready for
    torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)).

    A class with zero train examples (shouldn't happen for crop/disease --
    both label spaces are fully shared across datasets, see
    05_modelling_decisions.md -- but guarded anyway) gets weight 0.0 rather
    than a divide-by-zero, since the loss can never see that class.
    """
    if mode == "none":
        return [1.0] * len(id_map)
    if mode not in ("inv_freq", "sqrt_inv_freq"):
        raise ValueError(f"Unknown class_weight_mode {mode!r}. Must be 'none', 'inv_freq', or 'sqrt_inv_freq'.")

    counts = train_df[column].value_counts().to_dict()
    n_total = len(train_df)
    n_classes = len(id_map)

    weights = [0.0] * n_classes
    for name, idx in id_map.items():
        n = counts.get(name, 0)
        if n == 0:
            weights[idx] = 0.0
            continue
        inv_freq = n_total / (n_classes * n)
        weights[idx] = inv_freq ** 0.5 if mode == "sqrt_inv_freq" else inv_freq

    # Normalize so weights average to 1.0 -- keeps the loss magnitude
    # comparable across class_weight_mode choices and across crop_head vs
    # disease_head (different n_classes), rather than one mode silently
    # scaling the loss (and effectively the learning rate) up or down.
    nonzero = [w for w in weights if w > 0]
    if nonzero:
        mean_w = sum(nonzero) / len(nonzero)
        weights = [w / mean_w if w > 0 else 0.0 for w in weights]

    return weights
