#!/usr/bin/env python3
"""
LUMEN Phase 3 — Evaluation Module

Provides:
- Test-set evaluation
- Per-class metrics
- Confusion matrix
- Quadratic weighted kappa
- Classification report
- Prediction export
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import TrainingConfig
from .dataset import APTOSDataset
from .models import build_model

logger = logging.getLogger(__name__)

DR_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


class Evaluator:
    """Evaluates a trained model on the held-out test set."""

    def __init__(self, config: TrainingConfig, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config.resolve_device())
        self.model = build_model(config.model_name, config.num_classes, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference. Returns (true_labels, pred_labels, probabilities)."""
        all_labels = []
        all_preds = []
        all_probs = []
        all_ids = []

        for images, labels, img_ids in loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(img_ids)

        return (np.array(all_labels), np.array(all_preds),
                np.array(all_probs), all_ids)

    def evaluate(self, split: str = "test") -> Dict:
        """Run full evaluation on specified split."""
        if split == "test":
            csv_path = self.config.test_csv
        elif split == "val":
            csv_path = self.config.val_csv
        else:
            csv_path = self.config.train_csv

        ds = APTOSDataset(csv_path, self.config.img_dir, split=split,
                          target_size=self.config.input_size)
        loader = DataLoader(ds, batch_size=self.config.batch_size,
                            shuffle=False, num_workers=self.config.num_workers)

        true_labels, pred_labels, probabilities, img_ids = self._predict(loader)

        accuracy = accuracy_score(true_labels, pred_labels)
        macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        kappa = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')

        per_class = {}
        for cls in range(self.config.num_classes):
            cls_mask = true_labels == cls
            if cls_mask.sum() > 0:
                cls_pred = pred_labels[cls_mask]
                per_class[DR_LABELS[cls]] = {
                    'precision': float(precision_score(true_labels, pred_labels, labels=[cls], average='micro', zero_division=0)),
                    'recall': float(recall_score(true_labels, pred_labels, labels=[cls], average='micro', zero_division=0)),
                    'f1': float(f1_score(true_labels, pred_labels, labels=[cls], average='micro', zero_division=0)),
                    'support': int(cls_mask.sum()),
                }

        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=range(self.config.num_classes))
        norm_conf_matrix = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)

        report_text = classification_report(
            true_labels, pred_labels,
            target_names=[DR_LABELS[i] for i in range(self.config.num_classes)],
            zero_division=0
        )

        referable_mask = true_labels >= 2
        referable_pred = pred_labels >= 2
        referable_recall = float(recall_score(referable_mask, referable_pred, zero_division=0))

        results = {
            'split': split,
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'quadratic_kappa': float(kappa),
            'per_class': per_class,
            'confusion_matrix': conf_matrix.tolist(),
            'normalized_confusion_matrix': norm_conf_matrix.tolist(),
            'classification_report': report_text,
            'referable_dr_recall': referable_recall,
            'total_samples': len(true_labels),
        }

        return results, true_labels, pred_labels, probabilities, img_ids

    def save_confusion_matrix(self, conf_matrix: np.ndarray, title: str,
                              save_path: str, normalized: bool = False):
        """Plot and save confusion matrix."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        labels = [DR_LABELS[i] for i in range(self.config.num_classes)]

        if normalized:
            cm = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)

        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def save_prediction_examples(self, true_labels, pred_labels, probabilities,
                                 img_ids, save_dir: str, num_examples: int = 20):
        """Save prediction examples (correct and incorrect) as reference."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        correct_mask = true_labels == pred_labels
        incorrect_mask = ~correct_mask

        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(incorrect_mask)[0]

        examples = {
            'correct_examples': [],
            'incorrect_examples': [],
        }

        for idx in correct_indices[:num_examples]:
            examples['correct_examples'].append({
                'image_id': img_ids[idx],
                'true_label': int(true_labels[idx]),
                'predicted_label': int(pred_labels[idx]),
                'confidence': float(probabilities[idx].max()),
                'probabilities': probabilities[idx].tolist(),
            })

        for idx in incorrect_indices[:num_examples]:
            examples['incorrect_examples'].append({
                'image_id': img_ids[idx],
                'true_label': int(true_labels[idx]),
                'predicted_label': int(pred_labels[idx]),
                'confidence': float(probabilities[idx].max()),
                'true_class_confidence': float(probabilities[idx][true_labels[idx]]),
                'probabilities': probabilities[idx].tolist(),
            })

        with open(os.path.join(save_dir, 'prediction_examples.json'), 'w') as f:
            json.dump(examples, f, indent=2)

        return examples
