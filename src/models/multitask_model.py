"""The multitask formulation: one shared backbone, two independent linear
heads (crop, disease). See assets/docs/05_modelling_decisions.md for why
this formulation was built first.
"""
from __future__ import annotations

import torch.nn as nn

from src.models.backbones import build_backbone


class MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_crops: int,
        num_diseases: int,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        self.crop_head = nn.Linear(feature_dim, num_crops)
        self.disease_head = nn.Linear(feature_dim, num_diseases)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.crop_head(feats), self.disease_head(feats)
