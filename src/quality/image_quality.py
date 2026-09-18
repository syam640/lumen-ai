#!/usr/bin/env python3
"""
LUMEN Phase 2 — Image Quality Analysis Module

Analyzes fundus image quality metrics:
- Sharpness/Laplacian variance
- Brightness
- Contrast
- Exposure
- Quality scoring

Optimized: computes grayscale and HSV once per image and reuses across metrics.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))
from dataset_manager import DatasetManager, DR_LABELS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.END}\n")


@dataclass
class QualityMetrics:
    """Container for image quality metrics."""
    image_id: str
    file_path: str
    file_size_kb: float
    width: int
    height: int
    laplacian_variance: float
    mean_brightness: float
    brightness_std: float
    contrast: float
    mean_hue: float
    saturation_mean: float
    darkness_ratio: float
    bright_ratio: float
    quality_score: float
    quality_label: str


class ImageQualityAnalyzer:
    """Analyzes quality of fundus images."""

    def __init__(self):
        self.manager = DatasetManager()
        self.metrics_list: List[QualityMetrics] = []

        # Engineering heuristics — NOT clinically validated thresholds
        self.thresholds = {
            'sharpness_good': 100.0,
            'sharpness_acceptable': 50.0,
            'brightness_min': 30.0,
            'brightness_max': 220.0,
            'contrast_min': 30.0,
            'darkness_ratio_max': 0.3,
            'bright_ratio_max': 0.1
        }

    def analyze_single_image(self, img_path: Path) -> Optional[QualityMetrics]:
        """Analyze quality of a single image. Computes grayscale and HSV once."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None

            h, w = img.shape[:2]
            file_size = img_path.stat().st_size / 1024

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(laplacian.var())

            brightness = float(np.mean(gray))
            brightness_std = float(np.std(gray))
            contrast = float(np.std(gray))

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue_mean = float(np.mean(hsv[:, :, 0]))
            sat_mean = float(np.mean(hsv[:, :, 1]))

            total_pixels = gray.size
            dark_ratio = float(np.sum(gray < 30) / total_pixels)
            bright_ratio = float(np.sum(gray > 225) / total_pixels)

            metrics_dict = {
                'laplacian_variance': sharpness,
                'mean_brightness': brightness,
                'contrast': contrast,
                'darkness_ratio': dark_ratio,
                'bright_ratio': bright_ratio
            }
            quality_score = self._calculate_quality_score(metrics_dict)
            quality_label = self._get_quality_label(quality_score)

            return QualityMetrics(
                image_id=img_path.stem,
                file_path=str(img_path),
                file_size_kb=round(file_size, 2),
                width=w,
                height=h,
                laplacian_variance=round(sharpness, 2),
                mean_brightness=round(brightness, 2),
                brightness_std=round(brightness_std, 2),
                contrast=round(contrast, 2),
                mean_hue=round(hue_mean, 2),
                saturation_mean=round(sat_mean, 2),
                darkness_ratio=round(dark_ratio, 4),
                bright_ratio=round(bright_ratio, 4),
                quality_score=quality_score,
                quality_label=quality_label
            )

        except Exception as e:
            logger.warning(f"Failed to analyze {img_path.name}: {e}")
            return None

    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate combined quality score (0-1, higher is better)."""
        score = 1.0

        if metrics['laplacian_variance'] < self.thresholds['sharpness_acceptable']:
            score *= 0.5
        elif metrics['laplacian_variance'] < self.thresholds['sharpness_good']:
            score *= 0.8

        brightness = metrics['mean_brightness']
        if brightness < self.thresholds['brightness_min'] or brightness > self.thresholds['brightness_max']:
            score *= 0.6

        if metrics['contrast'] < self.thresholds['contrast_min']:
            score *= 0.7

        if metrics['darkness_ratio'] > self.thresholds['darkness_ratio_max']:
            score *= 0.6
        if metrics['bright_ratio'] > self.thresholds['bright_ratio_max']:
            score *= 0.7

        return round(score, 3)

    def _get_quality_label(self, score: float) -> str:
        if score >= 0.8:
            return "GOOD"
        elif score >= 0.6:
            return "ACCEPTABLE"
        elif score >= 0.4:
            return "POOR"
        else:
            return "BAD"

    def analyze_dataset(self, sample_size: int = 500) -> pd.DataFrame:
        """Analyze quality of dataset images (sampled or full)."""
        print_header("[PHASE 2.3] IMAGE QUALITY ANALYSIS")

        df = self.manager.load_labels()
        if df is None:
            raise ValueError("Could not load labels")

        actual_sample = min(sample_size, len(df))
        if actual_sample < len(df):
            sample_ids = df['id_code'].sample(actual_sample, random_state=42).values
            print(f"Analyzing quality of {actual_sample} images (sampled from {len(df)})...")
        else:
            sample_ids = df['id_code'].values
            print(f"Analyzing quality of all {actual_sample} images...")

        self.metrics_list = []

        for idx, img_id in enumerate(sample_ids):
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                path = self.manager.train_dir / f"{img_id}{ext}"
                if path.exists():
                    img_path = path
                    break

            if img_path is None:
                continue

            metrics = self.analyze_single_image(img_path)
            if metrics:
                self.metrics_list.append(metrics)

            if (idx + 1) % 250 == 0 or (idx + 1) == actual_sample:
                logger.info(f"  Quality progress: {idx + 1}/{actual_sample} analyzed, "
                            f"{len(self.metrics_list)} successful")

        metrics_df = pd.DataFrame([asdict(m) for m in self.metrics_list])
        self._print_quality_summary(metrics_df)
        self.save_quality_report(metrics_df)

        return metrics_df

    def _print_quality_summary(self, df: pd.DataFrame) -> None:
        """Print quality analysis summary."""
        print_header("[PHASE 2.3] QUALITY ANALYSIS SUMMARY")

        total = len(df)
        if total == 0:
            print("No images analyzed.")
            return

        quality_counts = df['quality_label'].value_counts()
        print("Quality Distribution:")
        for label in ['GOOD', 'ACCEPTABLE', 'POOR', 'BAD']:
            count = quality_counts.get(label, 0)
            pct = (count / total) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

        print(f"\nNote: Quality thresholds are engineering heuristics, NOT clinically validated.")

        print(f"\nSharpness (Laplacian Variance):")
        print(f"  Mean: {df['laplacian_variance'].mean():.2f}")
        print(f"  Std: {df['laplacian_variance'].std():.2f}")
        print(f"  Min: {df['laplacian_variance'].min():.2f}")
        print(f"  Max: {df['laplacian_variance'].max():.2f}")

        print(f"\nBrightness:")
        print(f"  Mean: {df['mean_brightness'].mean():.2f}")
        print(f"  Std: {df['mean_brightness'].std():.2f}")

        print(f"\nContrast:")
        print(f"  Mean: {df['contrast'].mean():.2f}")
        print(f"  Std: {df['contrast'].std():.2f}")

        print(f"\nDark Ratio:")
        print(f"  Mean: {df['darkness_ratio'].mean():.4f}")

        print(f"\nBright Ratio:")
        print(f"  Mean: {df['bright_ratio'].mean():.4f}")

        poor_images = df[df['quality_label'].isin(['POOR', 'BAD'])]
        if len(poor_images) > 0:
            print(f"\n{len(poor_images)} images with POOR or BAD quality")
            print("  These may need review or exclusion during training")

    def save_quality_report(self, metrics_df: pd.DataFrame) -> Path:
        """Save quality analysis report."""
        output_dir = self.manager.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "image_quality_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images_analyzed': len(metrics_df),
            'quality_distribution': metrics_df['quality_label'].value_counts().to_dict(),
            'metrics_summary': {
                'sharpness': {
                    'mean': float(metrics_df['laplacian_variance'].mean()),
                    'std': float(metrics_df['laplacian_variance'].std())
                },
                'brightness': {
                    'mean': float(metrics_df['mean_brightness'].mean()),
                    'std': float(metrics_df['mean_brightness'].std())
                },
                'contrast': {
                    'mean': float(metrics_df['contrast'].mean()),
                    'std': float(metrics_df['contrast'].std())
                }
            },
            'thresholds': self.thresholds,
            'threshold_note': 'Engineering heuristics only — NOT clinically validated'
        }

        json_path = output_dir / "image_quality_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nQuality reports saved to:")
        print(f"  {csv_path}")
        print(f"  {json_path}")

        return csv_path

    def run_analysis(self, sample_size: int = 500) -> pd.DataFrame:
        """Run complete quality analysis pipeline."""
        metrics_df = self.analyze_dataset(sample_size)
        return metrics_df


def main():
    analyzer = ImageQualityAnalyzer()
    metrics_df = analyzer.run_analysis(sample_size=500)
    return 0


if __name__ == "__main__":
    sys.exit(main())
