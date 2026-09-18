#!/usr/bin/env python3
"""
LUMEN Phase 3 — Trainer

Provides:
- Stage A: frozen backbone, train head only
- Stage B: unfreeze backbone, fine-tune with smaller LR
- Early stopping
- Model checkpointing
- Training/validation logging
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .config import TrainingConfig
from .dataset import APTOSDataset
from .models import build_model
from .losses import build_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the full training loop with staged unfreezing."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.resolve_device())
        self.model = build_model(config.model_name, config.num_classes, config.pretrained)
        self.model.to(self.device)

        class_weights = torch.FloatTensor(config.class_weights).to(self.device)
        self.criterion = build_loss(config.loss_fn, class_weights, config.focal_gamma)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': [], 'epoch_time': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def _build_dataloaders(self) -> tuple:
        """Build train/val dataloaders."""
        train_ds = APTOSDataset(
            self.config.train_csv, self.config.img_dir,
            split="train", target_size=self.config.input_size
        )
        val_ds = APTOSDataset(
            self.config.val_csv, self.config.img_dir,
            split="val", target_size=self.config.input_size
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers,
            pin_memory=(self.device.type == 'cuda'), drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=(self.device.type == 'cuda')
        )
        return train_loader, val_loader, train_ds

    def _set_stage_a(self):
        """Stage A: freeze backbone, train head only."""
        if hasattr(self.model, 'get_backbone_params'):
            for p in self.model.get_backbone_params():
                p.requires_grad = False
            for p in self.model.get_head_params():
                p.requires_grad = True
            logger.info("Stage A: backbone frozen, training head only")
        else:
            logger.info("Stage A: full model training (no backbone to freeze)")

    def _set_stage_b(self, lr: float):
        """Stage B: unfreeze backbone, fine-tune all with smaller LR."""
        for p in self.model.parameters():
            p.requires_grad = True
        logger.info(f"Stage B: backbone unfrozen, fine-tuning with lr={lr}")

    def _train_one_epoch(self, loader: DataLoader) -> tuple:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple:
        """Validate. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total if total > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
        }
        path = Path(self.config.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"  New best model saved (val_loss={val_loss:.4f})")

    def _save_history(self):
        """Save training history to JSON."""
        path = Path(self.config.log_dir) / "training_history.json"
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train(self) -> Dict:
        """Execute the full staged training loop."""
        train_loader, val_loader, train_ds = self._build_dataloaders()

        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Train: {len(train_ds)} images, Val: {len(val_loader.dataset)} images")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Loss: {self.config.loss_fn}")

        total_start = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            if epoch <= self.config.freeze_backbone_epochs:
                self._set_stage_a()
                lr = self.config.lr_head
            else:
                if epoch == self.config.freeze_backbone_epochs + 1:
                    self._set_stage_b(self.config.lr_backbone)
                lr = self.config.lr_backbone

            param_groups = [{'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': lr}]
            if epoch == 1 or epoch == self.config.freeze_backbone_epochs + 1:
                self.optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)

            if self.config.scheduler == "cosine" and epoch == 1:
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs,
                    eta_min=lr * 0.01
                )
            elif self.config.scheduler == "step" and hasattr(self, 'optimizer'):
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=self.config.scheduler_step,
                    gamma=self.config.scheduler_gamma
                )

            train_loss, train_acc = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            stage = "A" if epoch <= self.config.freeze_backbone_epochs else "B"
            logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} [{stage}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"lr={current_lr:.6f} time={epoch_time:.1f}s"
            )

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, val_loss, is_best)

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={self.config.patience})")
                break

        total_time = time.time() - total_start
        self._save_history()

        result = {
            'total_epochs': epoch,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'total_time_seconds': round(total_time, 1),
            'device': str(self.device),
            'model': self.config.model_name,
            'loss': self.config.loss_fn,
        }

        logger.info(f"Training complete: {epoch} epochs in {total_time:.0f}s")
        logger.info(f"Best epoch: {self.best_epoch}, val_loss: {self.best_val_loss:.4f}")

        return result
