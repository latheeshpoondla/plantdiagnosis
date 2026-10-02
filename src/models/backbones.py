"""Backbone builders for the two models this project compares:
ConvNeXt-Tiny and Swin-Tiny, both via torchvision (no timm dependency --
torchvision 0.16+ already ships both with ImageNet-pretrained weights; see
configs/model_config.json's "backbones" section and
assets/docs/05_modelling_decisions.md).

Each builder strips the ImageNet 1000-way classification head and returns
the bare feature extractor plus its output feature dimension, so
src/models/multitask_model.py (or any future single-head formulation) can
attach its own head(s) on top.
"""
from __future__ import annotations

import torch.nn as nn

BACKBONE_NAMES = ("convnext_tiny", "swin_tiny")


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (feature_extractor, feature_dim). `feature_extractor(x)` on a
    (B, 3, H, W) batch returns a (B, feature_dim) tensor -- global pooling
    already included, no manual flatten/pool needed by the caller."""
    import torchvision.models as tvm

    if name == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.convnext_tiny(weights=weights)
        # model.classifier is Sequential(LayerNorm2d, Flatten, Linear(768, 1000)).
        # Replacing only the final Linear keeps the LayerNorm2d+Flatten (so the
        # output is already a flat (B, 768) feature vector) without needing to
        # reimplement global pooling ourselves.
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        return model, feature_dim

    if name == "swin_tiny":
        weights = tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.swin_t(weights=weights)
        # model.head is the final Linear(768, 1000); torchvision's SwinTransformer
        # already does avgpool+flatten internally before calling head(x).
        feature_dim = model.head.in_features
        model.head = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unknown backbone {name!r}. Must be one of {BACKBONE_NAMES}.")
