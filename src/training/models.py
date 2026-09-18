#!/usr/bin/env python3
"""
LUMEN Phase 3 — Model Architectures

Provides:
- BaselineCNN: simple CNN for smoke testing
- DRResNet50: ResNet50 transfer learning for DR grading
- DREfficientNet: EfficientNet-B0 transfer learning alternative
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    """Simple CNN baseline for end-to-end verification."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DRResNet50(nn.Module):
    """ResNet50 transfer learning for DR grading."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_backbone_params(self):
        """Parameters excluding the classification head."""
        return list(self.backbone.parameters())[:-2]

    def get_head_params(self):
        """Parameters of the classification head only."""
        return list(self.backbone.fc.parameters())


class DREfficientNet(nn.Module):
    """EfficientNet-B0 transfer learning for DR grading."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_backbone_params(self):
        """Parameters excluding the classification head."""
        return list(self.backbone.features.parameters()) + \
               list(self.backbone.avgpool.parameters())

    def get_head_params(self):
        """Parameters of the classification head only."""
        return list(self.backbone.classifier.parameters())


def build_model(model_name: str, num_classes: int = 5,
                pretrained: bool = True) -> nn.Module:
    """Factory function to build models by name."""
    if model_name == 'baseline_cnn':
        return BaselineCNN(num_classes=num_classes)
    elif model_name == 'resnet50':
        return DRResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        return DREfficientNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: baseline_cnn, resnet50, efficientnet_b0")
