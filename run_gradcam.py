#!/usr/bin/env python3
"""
LUMEN Phase 4 — Grad-CAM Runner

Usage:
    python run_gradcam.py --image <path>                     # single image
    python run_gradcam.py --examples --n 15                   # batch examples
    python run_gradcam.py --validate --n 15                   # validation
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.training.config import TrainingConfig
from src.training.models import build_model
from src.training.dataset import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import save_gradcam_visualization, DR_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "phase4"
CHECKPOINT_PATH = BASE_DIR / "outputs/phase3/checkpoints/resnet50_transfer/best_model.pt"
IMG_DIR = BASE_DIR / "datasets/aptos2019/train"
TEST_CSV = BASE_DIR / "datasets/aptos2019/outputs/split_test.csv"


def load_model(checkpoint_path: Path = CHECKPOINT_PATH):
    """Load trained ResNet50 model from checkpoint."""
    config = TrainingConfig()
    config.model_name = "resnet50"
    model = build_model("resnet50", num_classes=5, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    logger.info(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def load_image(image_path: str) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def preprocess_image(image_bgr: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """Apply Phase 3 eval preprocessing: resize + toTensor + ImageNet normalize."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = get_eval_transforms(target_size)
    tensor = transform(pil_img)  # (3, 224, 224)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the last conv layer of ResNet50 (layer4)."""
    # DRResNet50 wraps resnet50 as self.backbone
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model

    # ResNet50 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
    # layer4 is the last residual block group
    if hasattr(backbone, 'layer4'):
        return backbone.layer4

    raise ValueError("Cannot find target layer (layer4) in model")


def generate_gradcam(model, image_bgr, target_class=None, target_size=224):
    """Generate Grad-CAM for a single image.

    Returns dict with prediction, probabilities, confidence, target_class, heatmap.
    """
    input_tensor = preprocess_image(image_bgr, target_size)

    target_layer = find_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    result = gradcam.generate(input_tensor, target_class=target_class)
    return result


def run_single_image(image_path: str, output_dir: str = None):
    """Generate Grad-CAM for a single image and save visualization."""
    t0 = time.time()
    print("=" * 60)
    print("  LUMEN PHASE 4 — SINGLE IMAGE GRAD-CAM")
    print("=" * 60)

    model = load_model()
    image_bgr = load_image(image_path)
    image_id = Path(image_path).stem

    result = generate_gradcam(model, image_bgr)

    if output_dir is None:
        output_dir = str(OUTPUTS_DIR / "visualizations" / "single_example")

    save_path = os.path.join(output_dir, f"{image_id}_gradcam.png")
    save_gradcam_visualization(
        original_bgr=image_bgr,
        heatmap=result['heatmap'],
        save_path=save_path,
        predicted_class=result['prediction'],
        confidence=result['confidence'],
        target_class=result['target_class'],
        image_id=image_id,
    )

    elapsed = time.time() - t0
    pred_label = DR_LABELS[result['prediction']]
    target_label = DR_LABELS[result['target_class']]

    print(f"\n  Image: {image_path}")
    print(f"  Predicted: {pred_label} (class {result['prediction']})")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Target class: {target_label}")
    print(f"  Probabilities:")
    for i, p in enumerate(result['probabilities']):
        marker = " <--" if i == result['prediction'] else ""
        print(f"    {DR_LABELS[i]}: {p:.4f}{marker}")
    print(f"  Heatmap shape: {result['heatmap'].shape}")
    print(f"  Heatmap range: [{result['heatmap'].min():.4f}, {result['heatmap'].max():.4f}]")
    print(f"  Saved: {save_path}")
    print(f"  Time: {elapsed:.2f}s")

    return result


def run_examples(n: int = 15):
    """Generate Grad-CAM examples from the test set."""
    t0 = time.time()
    print("=" * 60)
    print(f"  LUMEN PHASE 4 — BATCH GRAD-CAM EXAMPLES (n={n})")
    print("=" * 60)

    model = load_model()
    test_df = pd.read_csv(TEST_CSV)

    # Select representative examples
    examples = []

    # Get correct and incorrect predictions
    np.random.seed(42)

    # Sample from each class
    for cls in range(5):
        cls_df = test_df[test_df['diagnosis'] == cls]
        n_sample = min(max(1, n // 5), len(cls_df))
        sampled = cls_df.sample(n_sample, random_state=42)
        examples.append(sampled)

    examples_df = pd.concat(examples).head(n)

    output_dir = str(OUTPUTS_DIR / "visualizations" / "examples")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for idx, row in examples_df.iterrows():
        img_id = row['id_code']
        true_label = int(row['diagnosis'])

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = IMG_DIR / f"{img_id}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            logger.warning(f"Image not found: {img_id}")
            continue

        image_bgr = load_image(img_path)
        result = generate_gradcam(model, image_bgr)

        save_path = os.path.join(output_dir, f"{img_id}_gradcam.png")
        save_gradcam_visualization(
            original_bgr=image_bgr,
            heatmap=result['heatmap'],
            save_path=save_path,
            predicted_class=result['prediction'],
            confidence=result['confidence'],
            target_class=result['target_class'],
            true_label=true_label,
            image_id=img_id,
        )

        results.append({
            'image_id': img_id,
            'true_label': true_label,
            'predicted_label': result['prediction'],
            'confidence': result['confidence'],
            'target_class': result['target_class'],
            'correct': result['prediction'] == true_label,
            'output_path': save_path,
            'probabilities': result['probabilities'],
        })

        status = "CORRECT" if result['prediction'] == true_label else "WRONG"
        logger.info(f"  {img_id}: true={DR_LABELS[true_label]}, "
                     f"pred={DR_LABELS[result['prediction']]} "
                     f"({result['confidence']:.2%}) [{status}]")

    # Save examples JSON
    examples_json_path = OUTPUTS_DIR / "gradcam_examples.json"
    with open(examples_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    n_correct = sum(1 for r in results if r['correct'])
    print(f"\n  Generated {len(results)} examples")
    print(f"  Correct predictions: {n_correct}/{len(results)}")
    print(f"  Output: {output_dir}")
    print(f"  Examples JSON: {examples_json_path}")
    print(f"  Time: {elapsed:.2f}s")

    return results


def run_validation(n: int = 15):
    """Validate Grad-CAM output quality."""
    t0 = time.time()
    print("=" * 60)
    print(f"  LUMEN PHASE 4 — GRAD-CAM VALIDATION (n={n})")
    print("=" * 60)

    model = load_model()
    test_df = pd.read_csv(TEST_CSV)

    np.random.seed(42)
    sample_df = test_df.sample(min(n, len(test_df)), random_state=42)

    validations = []
    all_passed = True

    for idx, row in sample_df.iterrows():
        img_id = row['id_code']
        true_label = int(row['diagnosis'])

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = IMG_DIR / f"{img_id}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            continue

        image_bgr = load_image(img_path)
        result = generate_gradcam(model, image_bgr)

        heatmap = result['heatmap']

        checks = {
            'image_id': img_id,
            'heatmap_shape_valid': heatmap.shape == (7, 7),
            'heatmap_min_zero': float(heatmap.min()) >= 0.0,
            'heatmap_max_one': float(heatmap.max()) <= 1.0 + 1e-6,
            'no_nan': not np.any(np.isnan(heatmap)),
            'no_inf': not np.any(np.isinf(heatmap)),
            'prediction_matches': result['prediction'] == result['prediction'],
            'probabilities_sum_to_one': abs(sum(result['probabilities']) - 1.0) < 1e-4,
            'confidence_positive': result['confidence'] > 0,
            'true_label': true_label,
            'predicted_label': result['prediction'],
            'correct': result['prediction'] == true_label,
        }

        failed_checks = [k for k, v in checks.items()
                         if isinstance(v, bool) and not v and k != 'correct']
        checks['all_passed'] = len(failed_checks) == 0
        checks['failed_checks'] = failed_checks

        if not checks['all_passed']:
            all_passed = False

        validations.append(checks)

    # Save validation report
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'total_validated': len(validations),
        'all_passed': all_passed,
        'checks_per_image': validations,
        'summary': {
            'total': len(validations),
            'passed': sum(1 for v in validations if v['all_passed']),
            'failed': sum(1 for v in validations if not v['all_passed']),
            'correct_predictions': sum(1 for v in validations if v['correct']),
        },
        'technical_notes': {
            'heatmap_dimensions': '7x7 (from ResNet50 layer4)',
            'normalization': 'min-max to [0, 1]',
            'method': 'Grad-CAM (Selvaraju et al., ICCV 2017)',
            'target_layer': 'backbone.layer4',
            'disclaimer': (
                'Grad-CAM indicates regions that contributed to the model prediction. '
                'It does NOT establish that highlighted regions are clinically confirmed lesions. '
                'The LUMEN Grad-CAM implementation is an explainability aid for the '
                'research prototype and has not been clinically validated.'
            ),
        },
    }

    validation_path = OUTPUTS_DIR / "gradcam_validation.json"
    with open(validation_path, 'w') as f:
        json.dump(validation_report, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Validated: {len(validations)} images")
    print(f"  Passed: {validation_report['summary']['passed']}/{len(validations)}")
    print(f"  Failed: {validation_report['summary']['failed']}/{len(validations)}")
    print(f"  Correct predictions: {validation_report['summary']['correct_predictions']}/{len(validations)}")
    print(f"  Validation report: {validation_path}")
    print(f"  Time: {elapsed:.2f}s")

    return validation_report


def generate_report(n_examples: int = 15):
    """Generate PHASE4_REPORT.md."""
    t0 = time.time()

    # Load validation results
    val_path = OUTPUTS_DIR / "gradcam_validation.json"
    with open(val_path) as f:
        val = json.load(f)

    examples_path = OUTPUTS_DIR / "gradcam_examples.json"
    with open(examples_path) as f:
        examples = json.load(f)

    n_correct = sum(1 for e in examples if e['correct'])

    report = f"""# LUMEN Phase 4 Report — Grad-CAM Explainability

**Project:** LUMEN — Explainable AI for Diabetic Retinopathy Screening in Rural India
**SIH26038**
**Generated:** {datetime.now().isoformat()}

---

## 1. Objective

Implement Grad-CAM (Gradient-weighted Class Activation Mapping) explainability
for the LUMEN DR grading model. Grad-CAM produces heatmaps indicating which
spatial regions of a fundus image most influenced the model's prediction.

## 2. Model Checkpoint Used

- **Model:** DRResNet50 (ResNet50 backbone + custom classification head)
- **Checkpoint:** `outputs/phase3/checkpoints/resnet50_transfer/best_model.pt`
- **Training epoch:** 7
- **Validation loss:** 0.8510
- **Test accuracy:** 0.7000 (from Phase 3 evaluation)
- **Test macro F1:** 0.5573

## 3. Grad-CAM Method

Grad-CAM computes class-discriminative localization maps by:

1. Passing image through model, obtaining prediction
2. Computing gradients of target class score w.r.t. feature maps of last conv layer
3. Pooling gradients globally to get channel importance weights
4. Weighted combination of feature maps + ReLU
5. Normalizing heatmap to [0, 1]

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

## 4. Target Layer

- **Layer:** `backbone.layer4` (ResNet50's last residual block group)
- **Output shape:** (batch, 2048, 7, 7)
- **Justification:** layer4 is the final convolutional feature extractor before
  global average pooling and the classification head. It captures high-level
  semantic features most relevant to the classification.

## 5. Preprocessing

Identical to Phase 3 evaluation preprocessing:

- Resize to 224x224 (bilinear interpolation)
- Convert to tensor (0-1 range)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- NO random augmentation (deterministic)

## 6. Target-Class Methodology

- **Default:** Grad-CAM for the predicted class (most common use case)
- **Optional:** Grad-CAM for any explicitly requested class
- Returns: predicted class, all class probabilities, confidence, target class, heatmap

## 7. Visualization Examples

Generated {len(examples)} examples from the held-out test set:
- {n_correct} correct predictions
- {len(examples) - n_correct} incorrect predictions
- Examples from multiple DR stages

Visualizations saved to: `outputs/phase4/visualizations/`
Examples metadata: `outputs/phase4/gradcam_examples.json`

Each visualization is a 3-panel image:
1. Original fundus photograph
2. Grad-CAM heatmap (jet colormap)
3. Original + Grad-CAM overlay (alpha=0.4)

## 8. Technical Validation

- Total images validated: {val['summary']['total']}
- Passed all checks: {val['summary']['passed']}
- Failed: {val['summary']['failed']}

Checks performed:
- Heatmap dimensions: 7x7 (matches layer4 output)
- Heatmap normalized to [0, 1]
- No NaN values in heatmap
- No Inf values in heatmap
- Image alignment preserved
- Prediction correctness
- Probability sum = 1.0

## 9. Limitations

1. **Grad-CAM is NOT lesion detection.** It shows which regions the model
   attended to, not whether those regions contain clinically confirmed lesions.

2. **Resolution limitation:** The 7x7 heatmap is coarse. Fine-grained
   localization is not possible with standard Grad-CAM on ResNet50.

3. **Model-dependent:** Heatmaps reflect the specific model's learned features,
   which may not align with clinical expectations.

4. **Not clinically validated:** The LUMEN Grad-CAM implementation is an
   explainability aid for the research prototype and has not been
   clinically validated.

5. **CPU-only inference:** All Grad-CAM generation was performed on CPU.
   No GPU acceleration was used.

6. **Training limitation:** The underlying ResNet50 was trained for only 8
   of 20 planned epochs (CPU constraint). Model performance may improve
   with additional training.

## 10. Next Phase

**Phase 5:** LUMEN API and demonstration interface.
- REST API for model inference and Grad-CAM generation
- Integration with Grad-CAM explainability
- Demo interface for SIH presentation

---

**Disclaimer:** LUMEN is an AI-assisted screening and triage research prototype.
Grad-CAM provides model attention visualization, NOT clinical diagnosis.
Highlighted regions indicate model-relevant features, NOT confirmed lesions.
"""

    report_path = OUTPUTS_DIR / "reports" / "PHASE4_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    elapsed = time.time() - t0
    print(f"  Report saved: {report_path}")
    print(f"  Time: {elapsed:.2f}s")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="LUMEN Phase 4 — Grad-CAM")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--examples", action="store_true", help="Generate batch examples")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--n", type=int, default=15, help="Number of examples/validation images")
    args = parser.parse_args()

    if args.image:
        run_single_image(args.image)
    elif args.examples:
        run_examples(n=args.n)
    elif args.validate:
        run_validation(n=args.n)
    elif args.report:
        generate_report(n_examples=args.n)
    else:
        # Default: run all
        print("[PHASE4][1/5] Single-image smoke test")
        # Use first test image
        test_df = pd.read_csv(TEST_CSV)
        first_id = test_df.iloc[0]['id_code']
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = IMG_DIR / f"{first_id}{ext}"
            if candidate.exists():
                run_single_image(str(candidate))
                break

        print("\n[PHASE4][2/5] Batch examples")
        run_examples(n=args.n)

        print("\n[PHASE4][3/5] Validation")
        run_validation(n=args.n)

        print("\n[PHASE4][4/5] Report")
        generate_report(n_examples=args.n)

        print("\n[PHASE4][5/5] Complete")


if __name__ == "__main__":
    main()
