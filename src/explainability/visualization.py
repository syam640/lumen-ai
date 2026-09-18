#!/usr/bin/env python3
"""
LUMEN Phase 4 — Grad-CAM Visualization

Creates 3-panel visualization:
  Panel 1: Original fundus image
  Panel 2: Grad-CAM heatmap
  Panel 3: Original + Grad-CAM overlay
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DR_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def create_overlay(original_bgr: np.ndarray, heatmap: np.ndarray,
                   alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image.

    Args:
        original_bgr: Original image, shape (H, W, 3), BGR, uint8.
        heatmap: Heatmap, shape (h, w), float [0, 1].
        alpha: Overlay transparency.

    Returns:
        Overlay image, shape (H, W, 3), BGR, uint8.
    """
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_gradcam_visualization(
    original_bgr: np.ndarray,
    heatmap: np.ndarray,
    save_path: str,
    predicted_class: int = None,
    confidence: float = None,
    target_class: int = None,
    true_label: int = None,
    image_id: str = None,
    dpi: int = 150,
):
    """Save a 3-panel Grad-CAM visualization.

    Args:
        original_bgr: Original image (H, W, 3), BGR, uint8.
        heatmap: Grad-CAM heatmap (h, w), float [0, 1].
        save_path: Output file path.
        predicted_class: Predicted class index.
        confidence: Prediction confidence.
        target_class: Target class for heatmap.
        true_label: Ground truth label (optional).
        image_id: Image identifier (optional).
        dpi: Output resolution.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    overlay = create_overlay(original_bgr, heatmap)

    # Convert BGR to RGB for matplotlib
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original Fundus", fontsize=11)
    axes[0].axis('off')

    # Panel 2: Heatmap
    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis('off')

    # Panel 3: Overlay
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Model Attention Overlay", fontsize=11)
    axes[2].axis('off')

    # Title
    title_parts = []
    if image_id:
        title_parts.append(f"Image: {image_id}")
    if predicted_class is not None:
        label = DR_LABELS.get(predicted_class, str(predicted_class))
        title_parts.append(f"Predicted: {label}")
    if true_label is not None:
        label = DR_LABELS.get(true_label, str(true_label))
        title_parts.append(f"True: {label}")
    if confidence is not None:
        title_parts.append(f"Confidence: {confidence:.2%}")
    if target_class is not None and target_class != predicted_class:
        label = DR_LABELS.get(target_class, str(target_class))
        title_parts.append(f"Target: {label}")

    title = " | ".join(title_parts)
    fig.suptitle(title, fontsize=10, y=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
