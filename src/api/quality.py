#!/usr/bin/env python3
"""
LUMEN Phase 5 — Quality Checking Service

Wraps Phase 2 image quality analysis for single-image API use.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Engineering heuristics — NOT clinically validated thresholds
THRESHOLDS = {
    'sharpness_good': 100.0,
    'sharpness_acceptable': 50.0,
    'brightness_min': 30.0,
    'brightness_max': 220.0,
    'contrast_min': 30.0,
}


def analyze_image_quality(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Analyze quality of a BGR numpy image.

    Args:
        image_bgr: Image as BGR numpy array (H, W, 3).

    Returns:
        Dict with quality status, metrics, and message.
    """
    h, w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Determine quality status
    status = "GOOD"
    message = "Image quality is sufficient for screening."

    if sharpness < THRESHOLDS['sharpness_acceptable']:
        status = "POOR"
        message = "Image appears blurry. Please capture a sharper fundus image."
    elif brightness < THRESHOLDS['brightness_min']:
        status = "POOR"
        message = "Image is too dark. Please improve lighting conditions."
    elif brightness > THRESHOLDS['brightness_max']:
        status = "POOR"
        message = "Image is overexposed. Please reduce light intensity."
    elif contrast < THRESHOLDS['contrast_min']:
        status = "ACCEPTABLE"
        message = "Image has low contrast. Results may be less reliable."
    elif sharpness < THRESHOLDS['sharpness_good']:
        status = "ACCEPTABLE"
        message = "Image sharpness is moderate. Results may be less reliable."

    return {
        "status": status,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "width": w,
        "height": h,
        "message": message,
    }


def check_quality_for_inference(quality_result: Dict[str, Any]) -> bool:
    """Check if image quality is sufficient to proceed with inference.

    Returns True if quality is acceptable, False if too poor.
    """
    return quality_result["status"] in ("GOOD", "ACCEPTABLE")
