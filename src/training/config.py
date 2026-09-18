#!/usr/bin/env python3
"""
LUMEN Phase 3 — Training Configuration

All hyperparameters and paths in a single dataclass.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Experiment
    experiment_id: str = "exp001"
    description: str = ""

    # Model
    model_name: str = "resnet50"  # baseline_cnn, resnet50, efficientnet_b0
    num_classes: int = 5
    pretrained: bool = True
    input_size: int = 224

    # Data
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    img_dir: str = ""
    random_seed: int = 42

    # Training
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 30
    patience: int = 7  # early stopping patience

    # Optimizer
    optimizer: str = "adam"
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4

    # Loss
    loss_fn: str = "weighted_ce"  # weighted_ce, focal, ce
    focal_gamma: float = 2.0

    # Class weights (from Phase 2)
    class_weights: List[float] = field(default_factory=lambda: [
        0.2159, 1.0534, 0.3901, 2.0194, 1.3212
    ])

    # LR scheduler
    scheduler: str = "cosine"  # cosine, step, none
    scheduler_step: int = 10
    scheduler_gamma: float = 0.1

    # Staged training
    freeze_backbone_epochs: int = 5  # Stage A: freeze backbone

    # Augmentation
    augmentation: str = "standard"  # standard, none

    # Output paths
    checkpoint_dir: str = ""
    log_dir: str = ""
    plot_dir: str = ""
    metric_dir: str = ""

    # Device
    device: str = "auto"  # auto, cpu, cuda

    def save(self, path: str):
        """Save config to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def resolve_device(self) -> str:
        """Determine device."""
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_defaults(self, base_dir: str) -> 'TrainingConfig':
        """Fill in default paths based on base directory."""
        base = Path(base_dir)
        self.train_csv = str(base / "datasets/aptos2019/outputs/split_train.csv")
        self.val_csv = str(base / "datasets/aptos2019/outputs/split_val.csv")
        self.test_csv = str(base / "datasets/aptos2019/outputs/split_test.csv")
        self.img_dir = str(base / "datasets/aptos2019/train")
        self.checkpoint_dir = str(base / f"outputs/phase3/checkpoints/{self.experiment_id}")
        self.log_dir = str(base / f"outputs/phase3/logs/{self.experiment_id}")
        self.plot_dir = str(base / f"outputs/phase3/plots/{self.experiment_id}")
        self.metric_dir = str(base / f"outputs/phase3/metrics/{self.experiment_id}")
        return self
