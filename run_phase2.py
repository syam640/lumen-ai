#!/usr/bin/env python3
"""
LUMEN Phase 2 — Dataset Pipeline

Complete pipeline for dataset preparation:
1. Data audit (ALL 3662 images)
2. Image quality analysis
3. Preprocessing setup
4. Stratified splitting
5. Class imbalance strategy
6. Pipeline report

NO neural network training occurs in Phase 2.

Usage:
    python run_phase2.py              # full pipeline
    python run_phase2.py --smoke      # smoke test on 50 images
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))
from dataset_manager import DatasetManager, DR_LABELS

from src.data_audit import DataAuditor
from src.quality.image_quality import ImageQualityAnalyzer
from src.preprocessing.image_preprocessor import ImagePreprocessor, PreprocessingConfig
from src.splitting.stratified_splitter import StratifiedSplitter
from src.augmentation.class_imbalance import ClassImbalanceHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_header(text):
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.END}\n")


def print_step(step, total, text, status="IN PROGRESS"):
    status_color = Colors.YELLOW if status == "IN PROGRESS" else (
        Colors.GREEN if status == "PASS" else Colors.RED
    )
    print(f"  [{step}/{total}] {text:<45} {status_color}{status}{Colors.END}")


class DatasetPipeline:
    """Complete dataset preparation pipeline."""

    def __init__(self, smoke_test: bool = False):
        self.manager = DatasetManager()
        self.pipeline_results = {}
        self.smoke_test = smoke_test
        self.smoke_sample = 50 if smoke_test else None
        self.start_time = None

    def run_data_audit(self):
        """Step 1: Run data audit."""
        step = 1
        total = 6
        print_step(step, total, "[PHASE 2.1-2.3] Data audit", "IN PROGRESS")
        t0 = time.time()
        try:
            auditor = DataAuditor()
            report = auditor.run_audit()
            self.pipeline_results["data_audit"] = report["summary"]
            elapsed = time.time() - t0
            print_step(step, total, f"[PHASE 2.1-2.3] Data audit ({elapsed:.0f}s)", "PASS")
            return report
        except Exception as e:
            print_step(step, total, "[PHASE 2.1-2.3] Data audit", "FAIL")
            logger.error(f"Data audit failed: {e}")
            raise

    def run_quality_analysis(self):
        """Step 2: Run image quality analysis."""
        step = 2
        total = 6
        sample = self.smoke_sample or 500
        print_step(step, total, f"[PHASE 2.3] Quality analysis (n={sample})", "IN PROGRESS")
        t0 = time.time()
        try:
            analyzer = ImageQualityAnalyzer()
            metrics_df = analyzer.run_analysis(sample_size=sample)
            quality_dist = metrics_df["quality_label"].value_counts().to_dict()
            self.pipeline_results["quality_analysis"] = {
                "total_analyzed": len(metrics_df),
                "quality_distribution": quality_dist
            }
            elapsed = time.time() - t0
            print_step(step, total,
                       f"[PHASE 2.3] Quality analysis ({elapsed:.0f}s)", "PASS")
            return {"metrics": metrics_df, "distribution": quality_dist}
        except Exception as e:
            print_step(step, total, "[PHASE 2.3] Quality analysis", "FAIL")
            logger.error(f"Quality analysis failed: {e}")
            raise

    def run_preprocessing_setup(self):
        """Step 3: Setup preprocessing pipeline."""
        step = 3
        total = 6
        print_step(step, total, "[PHASE 2.4] Preprocessing setup", "IN PROGRESS")
        t0 = time.time()
        try:
            config = PreprocessingConfig(
                target_size=(224, 224),
                normalize=True,
                normalization_type="imagenet",
                apply_clahe=False,
                preserve_aspect_ratio=True
            )
            preprocessor = ImagePreprocessor(config)
            config_path = self.manager.outputs_dir / "preprocessing_config.json"
            preprocessor.save_config(config_path)

            test_img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = self.manager.train_dir / f"000c1434d8d7{ext}"
                if candidate.exists():
                    test_img_path = candidate
                    break
            preprocess_ok = False
            if test_img_path:
                result = preprocessor.preprocess_and_save(
                    test_img_path,
                    self.manager.outputs_dir / "preprocessing_smoke_test.png"
                )
                preprocess_ok = result

            self.pipeline_results["preprocessing"] = {
                "target_size": list(config.target_size),
                "normalization": config.normalization_type,
                "config_path": str(config_path),
                "smoke_test_ok": preprocess_ok
            }
            elapsed = time.time() - t0
            status = "PASS" if preprocess_ok else "FAIL"
            print_step(step, total,
                       f"[PHASE 2.4] Preprocessing setup ({elapsed:.0f}s)", status)
            return {"config": config, "preprocessor": preprocessor}
        except Exception as e:
            print_step(step, total, "[PHASE 2.4] Preprocessing setup", "FAIL")
            logger.error(f"Preprocessing setup failed: {e}")
            raise

    def run_stratified_splitting(self):
        """Step 4: Create stratified splits."""
        step = 4
        total = 6
        print_step(step, total, "[PHASE 2.5] Stratified splitting", "IN PROGRESS")
        t0 = time.time()
        try:
            splitter = StratifiedSplitter(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42
            )
            df = splitter.load_labels()
            train_df, val_df, test_df = splitter.create_splits(df)
            verification = splitter.verify_splits(train_df, val_df, test_df, df)
            splitter.save_splits(train_df, val_df, test_df, verification)
            self.pipeline_results["splitting"] = {
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "verification_status": verification["status"]
            }
            elapsed = time.time() - t0
            print_step(step, total,
                       f"[PHASE 2.5] Stratified splitting ({elapsed:.0f}s)",
                       "PASS" if verification["status"] == "PASS" else "FAIL")
            return {
                "train_df": train_df,
                "val_df": val_df,
                "test_df": test_df,
                "verification": verification
            }
        except Exception as e:
            print_step(step, total, "[PHASE 2.5] Stratified splitting", "FAIL")
            logger.error(f"Splitting failed: {e}")
            raise

    def run_class_imbalance_analysis(self):
        """Step 5: Analyze and handle class imbalance."""
        step = 5
        total = 6
        print_step(step, total, "[PHASE 2.6] Class imbalance analysis", "IN PROGRESS")
        t0 = time.time()
        try:
            handler = ClassImbalanceHandler()
            df = self.manager.load_labels()
            analysis = handler.analyze_imbalance(df)
            class_weights = handler.compute_class_weights(df, method="balanced")
            focal_weights = handler.compute_focal_loss_weights(df, gamma=2.0)
            aug_strategy = handler.suggest_augmentation_strategy(df)
            handler.save_imbalance_report(analysis, class_weights, focal_weights, aug_strategy)
            self.pipeline_results["class_imbalance"] = {
                "imbalance_ratio": analysis["imbalance_ratio"],
                "class_weights": class_weights,
                "augmentation_strategy": aug_strategy
            }
            elapsed = time.time() - t0
            print_step(step, total,
                       f"[PHASE 2.6] Class imbalance analysis ({elapsed:.0f}s)", "PASS")
            return {
                "analysis": analysis,
                "class_weights": class_weights,
                "focal_weights": focal_weights,
                "aug_strategy": aug_strategy
            }
        except Exception as e:
            print_step(step, total, "[PHASE 2.6] Class imbalance analysis", "FAIL")
            logger.error(f"Class imbalance analysis failed: {e}")
            raise

    def generate_pipeline_report(self):
        """Step 6: Generate final pipeline report."""
        step = 6
        total = 6
        print_step(step, total, "[PHASE 2.7] Pipeline report generation", "IN PROGRESS")
        t0 = time.time()
        try:
            total_elapsed = time.time() - self.start_time if self.start_time else 0
            report = {
                "pipeline_timestamp": datetime.now().isoformat(),
                "dataset": "APTOS 2019 Blindness Detection",
                "pipeline_mode": "smoke_test" if self.smoke_test else "full",
                "total_runtime_seconds": round(total_elapsed, 1),
                "pipeline_results": self.pipeline_results,
                "configuration": {
                    "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                    "random_seed": 42,
                    "preprocessing": {
                        "target_size": [224, 224],
                        "normalization": "imagenet",
                        "preserve_aspect_ratio": True,
                        "note": "Raw images are never modified. Preprocessing is applied at training time only."
                    },
                    "quality_thresholds_note": "Engineering heuristics only — NOT clinically validated",
                    "augmentation_note": "Augmentation must be applied AFTER splitting to prevent leakage across train/val/test",
                    "patient_leakage_note": (
                        "Patient-level leakage cannot be fully assessed — "
                        "APTOS 2019 does not provide patient identifiers"
                    )
                },
                "status": "COMPLETE",
                "next_phase": "Phase 3: Baseline DR Model Training"
            }
            if "class_imbalance" in self.pipeline_results:
                imbalance_ratio = self.pipeline_results["class_imbalance"]["imbalance_ratio"]
                if imbalance_ratio > 10:
                    report["warnings"] = ["Severe class imbalance detected"]
            output_dir = self.manager.outputs_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "phase2_pipeline_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            elapsed = time.time() - t0
            print_step(step, total,
                       f"[PHASE 2.7] Pipeline report generation ({elapsed:.0f}s)", "PASS")
            return report_path
        except Exception as e:
            print_step(step, total, "[PHASE 2.7] Pipeline report generation", "FAIL")
            logger.error(f"Report generation failed: {e}")
            raise

    def print_final_summary(self, report_path):
        """Print final pipeline summary."""
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        print_header("PHASE 2 COMPLETE — DATASET PIPELINE SUMMARY")
        results = self.pipeline_results
        print(f"Dataset: APTOS 2019 Blindness Detection")
        print(f"Mode: {'SMOKE TEST (50 images)' if self.smoke_test else 'FULL (all images)'}")
        print(f"Status: COMPLETE")
        print(f"Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        if "data_audit" in results:
            audit = results["data_audit"]
            print(f"\nData Audit: {audit.get('status', 'UNKNOWN')}")
        if "quality_analysis" in results:
            qa = results["quality_analysis"]
            print(f"\nQuality Analysis: {qa['total_analyzed']} images")
            for label, count in qa.get("quality_distribution", {}).items():
                print(f"  {label}: {count}")
        if "splitting" in results:
            split = results["splitting"]
            print(f"\nSplits Created:")
            print(f"  Train: {split['train_size']} images")
            print(f"  Val:   {split['val_size']} images")
            print(f"  Test:  {split['test_size']} images")
            print(f"  Verification: {split['verification_status']}")
        if "class_imbalance" in results:
            imbalance = results["class_imbalance"]
            print(f"\nClass Imbalance:")
            print(f"  Ratio: {imbalance['imbalance_ratio']}:1")
            print(f"  Class weights computed: Yes")
        if "preprocessing" in results:
            pp = results["preprocessing"]
            print(f"\nPreprocessing:")
            print(f"  Target size: {pp['target_size']}")
            print(f"  Normalization: {pp['normalization']}")
            print(f"  Smoke test: {'PASS' if pp.get('smoke_test_ok') else 'FAIL'}")
        print(f"\nReport: {report_path}")
        print(f"\nOutputs:")
        output_dir = self.manager.outputs_dir
        for f in sorted(output_dir.glob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")

    def run_full_pipeline(self):
        """Run the complete Phase 2 pipeline."""
        self.start_time = time.time()
        mode = "SMOKE TEST" if self.smoke_test else "FULL PIPELINE"
        print_header(f"LUMEN — PHASE 2: DATASET PIPELINE ({mode})")

        # Step 1: Data audit
        self.run_data_audit()

        # Step 2: Quality analysis
        self.run_quality_analysis()

        # Step 3: Preprocessing setup
        self.run_preprocessing_setup()

        # Step 4: Stratified splitting
        self.run_stratified_splitting()

        # Step 5: Class imbalance
        self.run_class_imbalance_analysis()

        # Step 6: Generate report
        report_path = self.generate_pipeline_report()

        # Print summary
        self.print_final_summary(report_path)

        return report_path


def main():
    parser = argparse.ArgumentParser(description="LUMEN Phase 2 Pipeline")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test on 50 images only")
    args = parser.parse_args()

    pipeline = DatasetPipeline(smoke_test=args.smoke)
    report_path = pipeline.run_full_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
