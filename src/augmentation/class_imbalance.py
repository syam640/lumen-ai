#!/usr/bin/env python3
"""
LUMEN Phase 2 — Class Imbalance Strategy Module

Addresses class imbalance in DR classification:
- Class weighting
- Oversampling (SMOTE-like for images)
- Undersampling
- Focal loss weights
- Augmentation-based balancing
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

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


class ClassImbalanceHandler:
    """Handles class imbalance in DR classification dataset."""
    
    def __init__(self):
        self.manager = DatasetManager()
        
    def analyze_imbalance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze class distribution and imbalance."""
        print_header("CLASS IMBALANCE ANALYSIS")
        
        # Count samples per class
        class_counts = df['diagnosis'].value_counts().sort_index()
        total = len(df)
        
        analysis = {
            'total_samples': total,
            'num_classes': len(class_counts),
            'class_counts': {},
            'class_percentages': {},
            'imbalance_ratio': 0.0,
            'minority_class': 0,
            'majority_class': 0,
            'minority_count': 0,
            'majority_count': 0,
            'effective_num_samples': 0
        }
        
        for stage, count in class_counts.items():
            analysis['class_counts'][int(stage)] = int(count)
            analysis['class_percentages'][int(stage)] = round((count / total) * 100, 2)
        
        # Calculate imbalance metrics
        min_class = class_counts.min()
        max_class = class_counts.max()
        
        analysis['imbalance_ratio'] = round(max_class / min_class, 2)
        analysis['minority_class'] = int(class_counts.idxmin())
        analysis['majority_class'] = int(class_counts.idxmax())
        analysis['minority_count'] = int(min_class)
        analysis['majority_count'] = int(max_class)
        
        # Effective number of samples (for focal loss)
        beta = 0.999
        effective_num = 1 - np.power(beta, class_counts.values)
        weights = (1 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * len(class_counts)
        analysis['effective_num_samples'] = float(np.mean(effective_num))
        
        # Print analysis
        print(f"Total samples: {total}")
        print(f"Number of classes: {len(class_counts)}")
        print(f"\nClass Distribution:")
        for stage in range(5):
            count = analysis['class_counts'].get(stage, 0)
            pct = analysis['class_percentages'].get(stage, 0)
            label = DR_LABELS[stage]
            print(f"  Stage {stage} ({label}): {count} ({pct}%)")
        
        print(f"\nImbalance Metrics:")
        print(f"  Imbalance Ratio: {analysis['imbalance_ratio']}:1")
        print(f"  Minority Class: Stage {analysis['minority_class']} ({analysis['minority_count']} samples)")
        print(f"  Majority Class: Stage {analysis['majority_class']} ({analysis['majority_count']} samples)")
        
        return analysis
    
    def compute_class_weights(self, df: pd.DataFrame, 
                             method: str = "balanced") -> Dict[int, float]:
        """
        Compute class weights for handling imbalance.
        
        Args:
            df: DataFrame with 'diagnosis' column
            method: 'balanced', 'sqrt', or 'inverse'
            
        Returns:
            Dictionary mapping class to weight
        """
        print_header("COMPUTING CLASS WEIGHTS")
        
        classes = df['diagnosis'].values
        unique_classes = np.unique(classes)
        
        if method == "balanced":
            # sklearn balanced weights
            weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=classes
            )
        elif method == "sqrt":
            # Square root of inverse frequency
            class_counts = Counter(classes)
            total = len(classes)
            weights = np.array([
                np.sqrt(total / class_counts[c]) 
                for c in unique_classes
            ])
        elif method == "inverse":
            # Inverse frequency
            class_counts = Counter(classes)
            total = len(classes)
            weights = np.array([
                total / class_counts[c] 
                for c in unique_classes
            ])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize weights
        weights = weights / weights.sum() * len(unique_classes)
        
        # Create dictionary
        class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}
        
        print(f"Method: {method}")
        print(f"\nClass Weights:")
        for stage in range(5):
            if stage in class_weights:
                weight = class_weights[stage]
                label = DR_LABELS[stage]
                print(f"  Stage {stage} ({label}): {weight:.4f}")
        
        return class_weights
    
    def compute_focal_loss_weights(self, df: pd.DataFrame, 
                                   gamma: float = 2.0) -> Dict[int, float]:
        """
        Compute weights for focal loss.
        
        Args:
            df: DataFrame with 'diagnosis' column
            gamma: Focal loss gamma parameter
            
        Returns:
            Dictionary mapping class to focal weight
        """
        print_header("COMPUTING FOCAL LOSS WEIGHTS")
        
        classes = df['diagnosis'].values
        class_counts = Counter(classes)
        total = len(classes)
        
        # Compute focal weights
        focal_weights = {}
        for stage in range(5):
            if stage in class_counts:
                # Weight = (1 - p_t)^gamma
                p_t = class_counts[stage] / total
                focal_weight = (1 - p_t) ** gamma
                focal_weights[stage] = float(focal_weight)
        
        # Normalize
        total_weight = sum(focal_weights.values())
        focal_weights = {k: v / total_weight * len(focal_weights) 
                        for k, v in focal_weights.items()}
        
        print(f"Gamma: {gamma}")
        print(f"\nFocal Loss Weights:")
        for stage in range(5):
            if stage in focal_weights:
                weight = focal_weights[stage]
                label = DR_LABELS[stage]
                print(f"  Stage {stage} ({label}): {weight:.4f}")
        
        return focal_weights
    
    def suggest_augmentation_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest augmentation strategy based on class distribution.
        
        Returns:
            Dictionary with augmentation recommendations
        """
        print_header("AUGMENTATION STRATEGY")
        
        class_counts = df['diagnosis'].value_counts().sort_index()
        max_count = class_counts.max()
        
        strategy = {
            'target_samples_per_class': int(max_count),
            'classes_to_augment': {},
            'augmentation_factors': {}
        }
        
        for stage, count in class_counts.items():
            stage = int(stage)
            if count < max_count:
                # Calculate how many augmented samples needed
                needed = max_count - count
                factor = max_count / count
                
                strategy['classes_to_augment'][stage] = {
                    'original_count': int(count),
                    'target_count': int(max_count),
                    'augmented_needed': int(needed),
                    'augmentation_factor': round(factor, 2)
                }
                strategy['augmentation_factors'][stage] = round(factor, 2)
        
        print(f"Target samples per class: {max_count}")
        print(f"\nClasses requiring augmentation:")
        for stage in range(5):
            if stage in strategy['classes_to_augment']:
                info = strategy['classes_to_augment'][stage]
                label = DR_LABELS[stage]
                print(f"  Stage {stage} ({label}):")
                print(f"    Original: {info['original_count']}")
                print(f"    Target: {info['target_count']}")
                print(f"    Factor: {info['augmentation_factor']}x")
        
        return strategy
    
    def save_imbalance_report(self, analysis: Dict[str, Any], 
                             class_weights: Dict[int, float],
                             focal_weights: Dict[int, float],
                             augmentation_strategy: Dict[str, Any]) -> Path:
        """Save imbalance analysis report."""
        output_dir = self.manager.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'class_weights': class_weights,
            'focal_loss_weights': focal_weights,
            'augmentation_strategy': augmentation_strategy,
            'recommendations': []
        }
        
        # Add recommendations
        if analysis['imbalance_ratio'] > 10:
            report['recommendations'].append(
                "Severe imbalance detected - consider oversampling minority classes"
            )
        if analysis['imbalance_ratio'] > 5:
            report['recommendations'].append(
                "Moderate imbalance - use class weights during training"
            )
        
        report_path = output_dir / "class_imbalance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        
        return report_path


def main():
    """Main entry point."""
    handler = ClassImbalanceHandler()
    
    # Load labels
    df = handler.manager.load_labels()
    if df is None:
        return 1
    
    # Analyze imbalance
    analysis = handler.analyze_imbalance(df)
    
    # Compute weights
    class_weights = handler.compute_class_weights(df, method="balanced")
    focal_weights = handler.compute_focal_loss_weights(df, gamma=2.0)
    
    # Get augmentation strategy
    aug_strategy = handler.suggest_augmentation_strategy(df)
    
    # Save report
    handler.save_imbalance_report(analysis, class_weights, focal_weights, aug_strategy)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())