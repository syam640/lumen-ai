#!/usr/bin/env python3
"""
LUMEN Phase 3 — APTOS Dataset for PyTorch

Provides:
- APTOSDataset class with split-aware loading
- Deterministic validation/test preprocessing
- Training-only augmentation
- ImageNet normalization
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(target_size: int = 224) -> T.Compose:
    """Training augmentation pipeline — randomized, non-deterministic."""
    return T.Compose([
        T.Resize((target_size, target_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(target_size: int = 224) -> T.Compose:
    """Deterministic validation/test preprocessing — NO random augmentation."""
    return T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class APTOSDataset(Dataset):
    """PyTorch Dataset for APTOS 2019 DR grading."""

    def __init__(self, csv_path: str, img_dir: str, split: str = "train",
                 target_size: int = 224):
        """
        Args:
            csv_path: Path to split CSV (columns: id_code, diagnosis)
            img_dir: Path to directory containing images
            split: 'train', 'val', or 'test'
            target_size: Resize target (square)
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.split = split
        self.target_size = target_size

        if split == "train":
            self.transform = get_train_transforms(target_size)
        else:
            self.transform = get_eval_transforms(target_size)

        self.image_ids = self.df['id_code'].values
        self.labels = self.df['diagnosis'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        label = int(self.labels[idx])

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = self.img_dir / f"{img_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_id}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label, img_id

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for this split's distribution."""
        class_counts = np.bincount(self.labels, minlength=5)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts + 1e-8)
        return torch.FloatTensor(weights)

    def get_label_distribution(self) -> Dict[int, int]:
        """Return class count dictionary."""
        return {int(k): int(v) for k, v in
                zip(*np.unique(self.labels, return_counts=True))}
