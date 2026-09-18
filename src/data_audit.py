#!/usr/bin/env python3
"""
LUMEN Phase 2 — Data Audit Script

Performs comprehensive data audit including:
- Dataset statistics
- Class distribution analysis
- Image integrity checks (all images)
- File hash duplicate detection
- Data integrity checks
- Leakage detection
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import Counter

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


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class DataAuditor:
    """Comprehensive data audit for APTOS 2019 dataset."""

    def __init__(self):
        self.manager = DatasetManager()
        self.audit_results = {}

    def load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata."""
        csv_path = self.manager.get_train_csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(f"train.csv not found at {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from train.csv")
        return df

    def audit_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit basic dataset statistics."""
        print_header("[PHASE 2.1] DATASET STATISTICS")

        stats = {
            'total_records': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'unique_ids': df['id_code'].nunique(),
            'duplicate_ids': int(df['id_code'].duplicated().sum()),
            'class_distribution': {},
            'class_percentages': {},
            'class_imbalance_ratio': 0.0,
            'minority_class': 0,
            'majority_class': 0,
            'minority_count': 0,
            'majority_count': 0
        }

        class_counts = df['diagnosis'].value_counts().sort_index()
        for stage, count in class_counts.items():
            stats['class_distribution'][int(stage)] = int(count)
            stats['class_percentages'][int(stage)] = round((count / len(df)) * 100, 2)

        min_class = class_counts.min()
        max_class = class_counts.max()
        stats['class_imbalance_ratio'] = round(max_class / min_class, 2)
        stats['minority_class'] = int(class_counts.idxmin())
        stats['majority_class'] = int(class_counts.idxmax())
        stats['minority_count'] = int(min_class)
        stats['majority_count'] = int(max_class)

        print(f"Total records: {stats['total_records']}")
        print(f"Unique IDs: {stats['unique_ids']}")
        print(f"Duplicate IDs: {stats['duplicate_ids']}")
        print(f"\nClass Distribution:")
        for stage in range(5):
            count = stats['class_distribution'].get(stage, 0)
            pct = stats['class_percentages'].get(stage, 0)
            label = DR_LABELS[stage]
            print(f"  Stage {stage} ({label}): {count} ({pct}%)")
        print(f"\nClass Imbalance Ratio: {stats['class_imbalance_ratio']}:1")
        print(f"Minority Class: Stage {stats['minority_class']} ({stats['minority_count']} samples)")
        print(f"Majority Class: Stage {stats['majority_class']} ({stats['majority_count']} samples)")

        self.audit_results['statistics'] = stats
        return stats

    def audit_image_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Audit ALL images for integrity: existence, readability, dimensions, file hashes.

        Reads each file once: bytes are used for both MD5 hash and cv2.imdecode.
        """
        print_header("[PHASE 2.2] IMAGE INTEGRITY AUDIT (ALL IMAGES)")

        results = {
            'total_images_checked': 0,
            'valid_images': 0,
            'corrupt_images': 0,
            'missing_images': 0,
            'wrong_format': 0,
            'zero_size': 0,
            'corrupt_ids': [],
            'missing_ids': [],
            'dimensions': [],
            'file_sizes_bytes': [],
            'file_hashes': {},
            'duplicate_hashes': {}
        }

        total = len(df)
        print(f"Checking all {total} images for existence and readability...")

        train_dir = self.manager.train_dir

        for idx, img_id in enumerate(df['id_code'].values):
            results['total_images_checked'] += 1

            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                path = train_dir / f"{img_id}{ext}"
                if path.exists():
                    img_path = path
                    break

            if img_path is None:
                results['missing_images'] += 1
                results['missing_ids'].append(img_id)
                if (idx + 1) % 250 == 0 or (idx + 1) == total:
                    logger.info(f"  Progress: {idx + 1}/{total} checked, "
                                f"{results['missing_images']} missing so far")
                continue

            file_size = img_path.stat().st_size
            if file_size == 0:
                results['zero_size'] += 1
                results['corrupt_ids'].append(img_id)
                continue

            results['file_sizes_bytes'].append(file_size)

            raw_bytes = img_path.read_bytes()
            file_hash = _md5_bytes(raw_bytes)
            if file_hash not in results['file_hashes']:
                results['file_hashes'][file_hash] = []
            results['file_hashes'][file_hash].append(img_id)

            try:
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    results['wrong_format'] += 1
                    results['corrupt_ids'].append(img_id)
                    continue

                h, w = img.shape[:2]
                results['dimensions'].append((w, h))
                results['valid_images'] += 1

            except Exception as e:
                results['corrupt_images'] += 1
                results['corrupt_ids'].append(img_id)

            if (idx + 1) % 250 == 0 or (idx + 1) == total:
                logger.info(f"  Progress: {idx + 1}/{total} checked, "
                            f"{results['valid_images']} valid, "
                            f"{results['missing_images']} missing, "
                            f"{results['corrupt_images'] + results['wrong_format'] + results['zero_size']} problematic")

        duplicate_hashes = {h: ids for h, ids in results['file_hashes'].items() if len(ids) > 1}
        results['duplicate_hashes'] = duplicate_hashes
        results['num_duplicate_hash_groups'] = len(duplicate_hashes)

        del results['file_hashes']

        if results['dimensions']:
            widths, heights = zip(*results['dimensions'])
            results['width_stats'] = {
                'min': int(min(widths)), 'max': int(max(widths)),
                'mean': round(float(np.mean(widths)), 1),
                'std': round(float(np.std(widths)), 1)
            }
            results['height_stats'] = {
                'min': int(min(heights)), 'max': int(max(heights)),
                'mean': round(float(np.mean(heights)), 1),
                'std': round(float(np.std(heights)), 1)
            }

        if results['file_sizes_bytes']:
            sizes_mb = [s / (1024 * 1024) for s in results['file_sizes_bytes']]
            results['file_size_stats'] = {
                'min_mb': round(min(sizes_mb), 2),
                'max_mb': round(max(sizes_mb), 2),
                'mean_mb': round(float(np.mean(sizes_mb)), 2),
                'std_mb': round(float(np.std(sizes_mb)), 2)
            }

        print(f"\nImage Integrity Results:")
        print(f"  Total checked: {results['total_images_checked']}")
        print(f"  Valid images: {results['valid_images']}")
        print(f"  Corrupt images: {results['corrupt_images']}")
        print(f"  Missing images: {results['missing_images']}")
        print(f"  Wrong format: {results['wrong_format']}")
        print(f"  Zero size: {results['zero_size']}")

        if 'width_stats' in results:
            print(f"\nDimension Statistics:")
            print(f"  Width:  {results['width_stats']['min']}-{results['width_stats']['max']} "
                  f"(mean={results['width_stats']['mean']})")
            print(f"  Height: {results['height_stats']['min']}-{results['height_stats']['max']} "
                  f"(mean={results['height_stats']['mean']})")

        if 'file_size_stats' in results:
            print(f"\nFile Size Statistics:")
            print(f"  Min: {results['file_size_stats']['min_mb']} MB")
            print(f"  Max: {results['file_size_stats']['max_mb']} MB")
            print(f"  Mean: {results['file_size_stats']['mean_mb']} MB")

        print(f"\nDuplicate file hash groups: {results['num_duplicate_hash_groups']}")
        if duplicate_hashes:
            for h, ids in list(duplicate_hashes.items())[:5]:
                print(f"  Hash {h[:12]}...: {len(ids)} files ({ids[:3]}...)")

        if results['corrupt_ids']:
            print(f"\nCorrupt image IDs (first 10): {results['corrupt_ids'][:10]}")
        if results['missing_ids']:
            print(f"\nMissing image IDs (first 10): {results['missing_ids'][:10]}")

        self.audit_results['image_integrity'] = results
        return results

    def audit_data_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for potential data leakage."""
        print_header("[PHASE 2.3] DATA LEAKAGE CHECKS")

        results = {
            'duplicate_ids': int(df['id_code'].duplicated().sum()),
            'duplicate_id_list': df[df['id_code'].duplicated()]['id_code'].tolist(),
            'id_pattern_consistent': True,
            'label_range_valid': True,
            'invalid_labels': [],
            'potential_leakage': False,
            'leakage_details': [],
            'patient_level_note': (
                "Patient-level leakage cannot be fully assessed from the available "
                "APTOS metadata. The APTOS 2019 dataset does not provide patient "
                "identifiers; each image ID is unique per examination. Multiple images "
                "from the same patient could exist across splits without detection."
            )
        }

        if results['duplicate_ids'] > 0:
            results['potential_leakage'] = True
            results['leakage_details'].append(
                f"Found {results['duplicate_ids']} duplicate IDs"
            )
            print(f"WARNING: Found {results['duplicate_ids']} duplicate IDs")
            print(f"  IDs: {results['duplicate_id_list'][:10]}")
        else:
            print("No duplicate IDs found")

        id_lengths = df['id_code'].str.len()
        if id_lengths.nunique() > 1:
            results['id_pattern_consistent'] = False
            results['leakage_details'].append("Inconsistent ID lengths")
            print(f"WARNING: Inconsistent ID lengths: {id_lengths.value_counts().to_dict()}")
        else:
            print(f"ID pattern consistent (length={id_lengths.iloc[0]})")

        valid_labels = set(DR_LABELS.keys())
        invalid_mask = ~df['diagnosis'].isin(valid_labels)
        if invalid_mask.any():
            results['label_range_valid'] = False
            results['invalid_labels'] = df[invalid_mask]['diagnosis'].unique().tolist()
            results['leakage_details'].append(f"Invalid labels: {results['invalid_labels']}")
            print(f"WARNING: Invalid labels found: {results['invalid_labels']}")
        else:
            print("All labels in valid range (0-4)")

        print(f"\nNote: {results['patient_level_note']}")

        if results['potential_leakage']:
            print(f"\nPotential leakage detected:")
            for detail in results['leakage_details']:
                print(f"  - {detail}")
        else:
            print(f"\nNo obvious data leakage detected")

        self.audit_results['leakage'] = results
        return results

    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'dataset': 'APTOS 2019 Blindness Detection',
            'audit_results': self.audit_results,
            'summary': {
                'status': 'PASS',
                'issues': [],
                'recommendations': []
            }
        }

        stats = self.audit_results.get('statistics', {})
        integrity = self.audit_results.get('image_integrity', {})
        leakage = self.audit_results.get('leakage', {})

        if stats.get('duplicate_ids', 0) > 0:
            report['summary']['issues'].append(f"Duplicate IDs: {stats['duplicate_ids']}")
        if integrity.get('corrupt_images', 0) > 0:
            report['summary']['issues'].append(f"Corrupt images: {integrity['corrupt_images']}")
        if integrity.get('missing_images', 0) > 0:
            report['summary']['issues'].append(f"Missing images: {integrity['missing_images']}")
        if integrity.get('num_duplicate_hash_groups', 0) > 0:
            report['summary']['issues'].append(
                f"Duplicate file hashes: {integrity['num_duplicate_hash_groups']} groups"
            )
        if leakage.get('potential_leakage', False):
            report['summary']['issues'].append("Potential data leakage detected")
        if stats.get('class_imbalance_ratio', 0) > 5:
            report['summary']['recommendations'].append(
                f"High class imbalance ({stats['class_imbalance_ratio']}:1) - "
                "consider weighted sampling or controlled augmentation"
            )

        if report['summary']['issues']:
            report['summary']['status'] = 'WARNING'

        output_dir = self.manager.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "data_audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print_header("[PHASE 2.7] AUDIT SUMMARY")
        print(f"Status: {report['summary']['status']}")
        if report['summary']['issues']:
            print("\nIssues:")
            for issue in report['summary']['issues']:
                print(f"  - {issue}")
        if report['summary']['recommendations']:
            print("\nRecommendations:")
            for rec in report['summary']['recommendations']:
                print(f"  -> {rec}")
        print(f"\nReport saved to: {report_path}")

        return report

    def run_audit(self) -> Dict[str, Any]:
        """Run complete data audit pipeline."""
        print_header("LUMEN — DATA AUDIT")

        df = self.load_metadata()

        self.audit_dataset_statistics(df)
        self.audit_image_integrity(df)
        self.audit_data_leakage(df)

        report = self.generate_audit_report()
        return report


def main():
    auditor = DataAuditor()
    report = auditor.run_audit()
    return 0 if report['summary']['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
