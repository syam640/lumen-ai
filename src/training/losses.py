#!/usr/bin/env python3
"""
LUMEN Phase 3 — Loss Functions

Provides:
- WeightedCrossEntropy: standard CE with class weights
- FocalLoss: focal loss for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropy(nn.Module):
    """Cross-entropy loss with class weights."""

    def __init__(self, class_weights: torch.Tensor = None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer('weight', class_weights)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """Focal loss: FL(pt) = -alpha_t * (1-pt)^gamma * log(pt)."""

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer('weight', class_weights)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_loss(loss_name: str, class_weights: torch.Tensor = None,
               gamma: float = 2.0) -> nn.Module:
    """Factory function to build loss by name."""
    loss_map = {
        'weighted_ce': lambda: WeightedCrossEntropy(class_weights),
        'focal': lambda: FocalLoss(gamma=gamma, class_weights=class_weights),
        'ce': lambda: nn.CrossEntropyLoss(),
    }
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from: {list(loss_map.keys())}")
    return loss_map[loss_name]()
