#!/usr/bin/env python3
"""
LUMEN Phase 5 — Demo Script

CLI fallback for SIH presentation if web UI fails.

Usage:
    python demo.py --image <fundus_image>
"""

import sys
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.api.inference import LUMENInferenceService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DR_LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def main():
    parser = argparse.ArgumentParser(description="LUMEN Demo — DR Screening")
    parser.add_argument("--image", required=True, help="Path to fundus image")
    parser.add_argument("--checkpoint",
                        default="outputs/phase3/checkpoints/resnet50_transfer/best_model.pt",
                        help="Model checkpoint path")
    parser.add_argument("--output-dir", default="outputs/phase5/demo",
                        help="Output directory for Grad-CAM")
    args = parser.parse_args()

    import cv2
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("  LUMEN — DR Screening Demo")
    print("  AI-assisted research prototype")
    print("=" * 60)

    # Load model
    t0 = time.time()
    service = LUMENInferenceService(args.checkpoint)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Read image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    # Run screening
    print(f"\n  Image: {image_path.name}")
    result = service.screen(
        image_bgr,
        generate_gradcam=True,
        output_dir=args.output_dir,
        image_id=image_path.stem,
    )

    # Display results
    print(f"\n  {'=' * 50}")
    print(f"  SCREENING RESULT")
    print(f"  {'=' * 50}")

    if not result['success']:
        print(f"\n  FAILED: {result['error']}")
        sys.exit(1)

    q = result['quality']
    print(f"\n  Image Quality: {q['status']}")
    print(f"    Brightness: {q['brightness']}")
    print(f"    Contrast:   {q['contrast']}")
    print(f"    Sharpness:  {q['sharpness']}")
    if q['message']:
        print(f"    Note: {q['message']}")

    p = result['prediction']
    print(f"\n  DR Stage: {p['stage']} — {p['label']}")
    print(f"  Confidence: {p['confidence']:.2%}")

    print(f"\n  Class Probabilities:")
    for stage_str, prob in result['probabilities'].items():
        stage = int(stage_str)
        label = DR_LABELS[stage]
        bar = "#" * int(prob * 40)
        marker = " <--" if stage == p['stage'] else ""
        print(f"    Stage {stage} ({label}): {prob:.4f} {bar}{marker}")

    t = result['triage']
    print(f"\n  Triage Priority: {t['priority']}")
    print(f"    {t['reason']}")

    e = result['explainability']
    print(f"\n  Explainability: {e['method']}")
    if e['heatmap_available']:
        vis_path = e.get('visualization_path', 'N/A')
        print(f"    Visualization: {vis_path}")
    else:
        print(f"    Grad-CAM not available")

    print(f"\n  Inference Time: {result['inference_time_ms']:.0f}ms")

    print(f"\n  {'=' * 50}")
    print(f"  DISCLAIMER")
    print(f"  {'=' * 50}")
    print(f"  LUMEN is an AI-assisted research prototype for screening support.")
    print(f"  It does not provide a definitive diagnosis and does not replace")
    print(f"  evaluation by a qualified eye-care professional.")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    main()
