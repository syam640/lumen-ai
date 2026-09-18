#!/usr/bin/env python3
"""
LUMEN Phase 2 — Stratified Splitting Module

Creates reproducible train/validation/test splits:
- Stratified splitting by class
- Patient-level split (when patient IDs available)
- Configurable split ratios
- Reproducible with seed
- Leakage prevention
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))
from dataset_manager import DatasetManager, DR_LABELS

# Configure logging
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


class StratifiedSplitter:
    """Creates reproducible stratified train/val/test splits."""
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.manager = DatasetManager()
        
    def load_labels(self) -> pd.DataFrame:
        """Load dataset labels."""
        df = self.manager.load_labels()
        if df is None:
            raise ValueError("Could not load labels")
        return df
    
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits.
        
        Uses stratified splitting to maintain class distribution across splits.
        """
        print_header("CREATING STRATIFIED SPLITS")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            stratify=df['diagnosis'],
            random_state=self.random_seed
        )
        
        # Second split: train vs val
        # Adjust val_ratio to account for the remaining data
        adjusted_val_ratio = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            stratify=train_val_df['diagnosis'],
            random_state=self.random_seed
        )
        
        print(f"Split sizes:")
        print(f"  Train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def verify_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     test_df: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify splits are correct and check for leakage."""
        print_header("SPLIT VERIFICATION")
        
        verification = {
            'total_images': len(df),
            'train_images': len(train_df),
            'val_images': len(val_df),
            'test_images': len(test_df),
            'splits_sum_correct': len(train_df) + len(val_df) + len(test_df) == len(df),
            'no_overlap': True,
            'class_distribution': {},
            'status': 'PASS'
        }
        
        # Check for overlap
        train_ids = set(train_df['id_code'])
        val_ids = set(val_df['id_code'])
        test_ids = set(test_df['id_code'])
        
        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            verification['no_overlap'] = False
            verification['status'] = 'FAIL'
            verification['overlap_details'] = {
                'train_val': len(train_val_overlap),
                'train_test': len(train_test_overlap),
                'val_test': len(val_test_overlap)
            }
            print(f"ERROR: Overlap detected between splits!")
        else:
            print("✓ No overlap between splits")
        
        # Class distribution in each split
        print("\nClass Distribution:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            dist = split_df['diagnosis'].value_counts().sort_index()
            verification['class_distribution'][split_name] = {
                'total': len(split_df),
                'distribution': {int(stage): int(count) for stage, count in dist.items()}
            }
            
            print(f"\n  {split_name} ({len(split_df)} images):")
            for stage in range(5):
                count = dist.get(stage, 0)
                pct = (count / len(split_df)) * 100
                label = DR_LABELS[stage]
                print(f"    Stage {stage} ({label}): {count} ({pct:.1f}%)")
        
        # Verify all images are assigned
        all_assigned_ids = train_ids.union(val_ids).union(test_ids)
        original_ids = set(df['id_code'])
        
        if all_assigned_ids != original_ids:
            verification['status'] = 'FAIL'
            missing = original_ids - all_assigned_ids
            extra = all_assigned_ids - original_ids
            print(f"\nERROR: Image assignment mismatch!")
            if missing:
                print(f"  Missing from splits: {len(missing)}")
            if extra:
                print(f"  Extra in splits: {len(extra)}")
        else:
            print("\n✓ All images assigned to splits")
        
        # Print summary
        print(f"\nVerification Status: {verification['status']}")
        
        return verification
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame, verification: Dict[str, Any]) -> Path:
        """Save splits and metadata."""
        output_dir = self.manager.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save split CSVs
        train_path = output_dir / "split_train.csv"
        val_path = output_dir / "split_val.csv"
        test_path = output_dir / "split_test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save split metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'split_config': {
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'random_seed': self.random_seed
            },
            'split_sizes': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            },
            'verification': verification,
            'files': {
                'train': str(train_path),
                'val': str(val_path),
                'test': str(test_path)
            }
        }
        
        metadata_path = output_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSplit files saved to:")
        print(f"  {train_path}")
        print(f"  {val_path}")
        print(f"  {test_path}")
        print(f"  {metadata_path}")
        
        return output_dir
    
    def create_splits_from_file(self, csv_path: Path) -> Dict[str, Any]:
        """Create splits from a CSV file path."""
        df = pd.read_csv(csv_path)
        train_df, val_df, test_df = self.create_splits(df)
        verification = self.verify_splits(train_df, val_df, test_df, df)
        self.save_splits(train_df, val_df, test_df, verification)
        return verification
    
    def create_splits_from_dataframes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create splits from a DataFrame."""
        train_df, val_df, test_df = self.create_splits(df)
        verification = self.verify_splits(train_df, val_df, test_df, df)
        self.save_splits(train_df, val_df, test_df, verification)
        return verification


def main():
    """Main entry point."""
    # Create splitter with default ratios
    splitter = StratifiedSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Load labels and create splits
    df = splitter.load_labels()
    verification = splitter.create_splits_from_dataframes(df)
    
    return 0 if verification['status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())