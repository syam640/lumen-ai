#!/usr/bin/env python3
"""
LUMEN Phase 5 — Inference Service

Wraps model loading, preprocessing, inference, and Grad-CAM
into a single reusable service.
"""

import io
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.models import build_model
from src.training.dataset import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import create_overlay
from src.api.quality import analyze_image_quality, check_quality_for_inference
from src.triage.rules import get_triage

logger = logging.getLogger(__name__)

DR_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


class LUMENInferenceService:
    """Single-model inference service for LUMEN screening."""

    def __init__(self, checkpoint_path: str):
        """Load model and prepare for inference.

        Args:
            checkpoint_path: Path to trained model checkpoint.
        """
        self.device = torch.device("cpu")
        logger.info(f"Loading model from {checkpoint_path}...")

        self.model = build_model("resnet50", num_classes=5, pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # Find target layer for Grad-CAM
        self.target_layer = self.model.backbone.layer4

        # Preprocessing transform
        self.transform = get_eval_transforms(224)

        logger.info(f"Model loaded. Checkpoint epoch={ckpt.get('epoch')}, "
                     f"val_loss={ckpt.get('val_loss'):.4f}")

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy image to preprocessed tensor."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img)
        return tensor.unsqueeze(0)

    def screen(self, image_bgr: np.ndarray,
               generate_gradcam: bool = True,
               output_dir: Optional[str] = None,
               image_id: str = "upload") -> Dict[str, Any]:
        """Run full screening pipeline on a single image.

        Args:
            image_bgr: Fundus image as BGR numpy array.
            generate_gradcam: Whether to generate Grad-CAM.
            output_dir: Directory to save Grad-CAM visualization.
            image_id: Identifier for this image.

        Returns:
            Complete screening result dict.
        """
        total_start = time.time()

        # 1. Quality check
        t0 = time.time()
        quality = analyze_image_quality(image_bgr)
        quality_time = (time.time() - t0) * 1000
        logger.info(f"[SCREEN] quality={quality['status']} ({quality_time:.0f}ms)")

        # If quality is BAD, skip inference
        if quality['status'] == "BAD":
            return {
                "success": False,
                "prediction": {"stage": -1, "label": "N/A", "confidence": 0.0},
                "probabilities": {},
                "quality": quality,
                "triage": {"priority": "N/A", "reason": "Image quality insufficient."},
                "explainability": {"method": "Grad-CAM", "target_class": -1,
                                   "heatmap_available": False},
                "inference_time_ms": (time.time() - total_start) * 1000,
                "error": quality["message"],
            }

        # 2. Preprocessing + inference
        t0 = time.time()
        input_tensor = self._preprocess(image_bgr)
        input_tensor.requires_grad_(False)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze(0)

        predicted_class = int(output.argmax(dim=1).item())
        confidence = float(probs[predicted_class].item())
        all_probs = {str(i): float(probs[i].item()) for i in range(5)}
        inference_time = (time.time() - t0) * 1000
        logger.info(f"[SCREEN] inference={DR_LABELS[predicted_class]} "
                     f"({confidence:.2%}) ({inference_time:.0f}ms)")

        # 3. Triage
        triage = get_triage(predicted_class, confidence)

        # 4. Grad-CAM
        gradcam_available = False
        vis_path = None
        t0 = time.time()

        if generate_gradcam:
            try:
                # Need gradients for Grad-CAM
                input_tensor_gc = self._preprocess(image_bgr)
                input_tensor_gc.requires_grad_(True)

                gradcam = GradCAM(self.model, self.target_layer)
                gc_result = gradcam.generate(input_tensor_gc, target_class=predicted_class)
                gradcam_available = True

                if output_dir:
                    from src.explainability.visualization import save_gradcam_visualization
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    vis_path = str(Path(output_dir) / f"{image_id}_gradcam.png")
                    save_gradcam_visualization(
                        original_bgr=image_bgr,
                        heatmap=gc_result['heatmap'],
                        save_path=vis_path,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        target_class=predicted_class,
                        image_id=image_id,
                    )
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")

        gradcam_time = (time.time() - t0) * 1000
        logger.info(f"[SCREEN] gradcam={'OK' if gradcam_available else 'FAIL'} "
                     f"({gradcam_time:.0f}ms)")

        total_time = (time.time() - total_start) * 1000
        logger.info(f"[SCREEN] total={total_time:.0f}ms")

        return {
            "success": True,
            "prediction": {
                "stage": predicted_class,
                "label": DR_LABELS[predicted_class],
                "confidence": confidence,
            },
            "probabilities": all_probs,
            "quality": quality,
            "triage": triage,
            "explainability": {
                "method": "Grad-CAM",
                "target_class": predicted_class,
                "heatmap_available": gradcam_available,
                "visualization_path": vis_path,
            },
            "inference_time_ms": total_time,
            "error": None,
        }
