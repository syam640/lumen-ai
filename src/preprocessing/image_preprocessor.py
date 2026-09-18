#!/usr/bin/env python3
"""
LUMEN Phase 2 — Image Preprocessing Module

Provides reproducible preprocessing pipeline:
- Image loading and validation
- Resize to target dimensions
- Normalization
- Color correction (optional)
- CLAHE enhancement (optional)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "datasets"))
from dataset_manager import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    target_size: Tuple[int, int] = (224, 224)  # (width, height)
    normalize: bool = True
    normalization_type: str = "imagenet"  # "imagenet", "minmax", "zscore"
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    apply_color_correction: bool = False
    preserve_aspect_ratio: bool = True
    padding_color: Tuple[int, int, int] = (0, 0, 0)  # Black padding
    output_format: str = "png"
    
    # ImageNet normalization values
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImagePreprocessor:
    """Provides reproducible image preprocessing."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.manager = DatasetManager()
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image from path."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            return img
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            return None
    
    def resize_with_padding(self, img: np.ndarray, 
                           target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while preserving aspect ratio with padding."""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), self.config.padding_color, dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def normalize_image(self, img: np.ndarray, 
                       normalization_type: str) -> np.ndarray:
        """Normalize image pixels."""
        img_float = img.astype(np.float32) / 255.0
        
        if normalization_type == "imagenet":
            # ImageNet normalization
            mean = np.array(self.config.imagenet_mean, dtype=np.float32)
            std = np.array(self.config.imagenet_std, dtype=np.float32)
            img_float = (img_float - mean) / std
        elif normalization_type == "minmax":
            # Min-max normalization (already 0-1 after /255)
            pass
        elif normalization_type == "zscore":
            # Z-score normalization
            mean = img_float.mean()
            std = img_float.std()
            img_float = (img_float - mean) / (std + 1e-8)
        
        return img_float
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline to image."""
        # 1. CLAHE enhancement (optional)
        if self.config.apply_clahe:
            img = self.apply_clahe(img)
        
        # 2. Resize with padding
        if self.config.preserve_aspect_ratio:
            img = self.resize_with_padding(img, self.config.target_size)
        else:
            img = cv2.resize(img, self.config.target_size, 
                           interpolation=cv2.INTER_LINEAR)
        
        # 3. Normalize
        if self.config.normalize:
            img = self.normalize_image(img, self.config.normalization_type)
        
        return img
    
    def preprocess_and_save(self, image_path: Path, output_path: Path) -> bool:
        """Preprocess single image and save."""
        img = self.load_image(image_path)
        if img is None:
            return False
        
        try:
            processed = self.preprocess_image(img)
            
            # Convert back to uint8 for saving
            if processed.dtype != np.uint8:
                if processed.min() < 0:
                    processed = ((processed - processed.min()) / 
                               (processed.max() - processed.min()) * 255)
                else:
                    processed = processed * 255
                processed = processed.astype(np.uint8)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save
            cv2.imwrite(str(output_path), processed)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to preprocess {image_path.name}: {e}")
            return False
    
    def batch_preprocess(self, image_paths: list, output_dir: Path,
                        prefix: str = "processed_") -> Dict[str, Any]:
        """Batch preprocess multiple images."""
        results = {
            'total': len(image_paths),
            'success': 0,
            'failed': 0,
            'failed_ids': []
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img_path in enumerate(image_paths):
            output_path = output_dir / f"{prefix}{img_path.name}"
            
            if self.preprocess_and_save(img_path, output_path):
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_ids'].append(img_path.stem)
            
            if (idx + 1) % 500 == 0:
                logger.info(f"Processed {idx + 1}/{len(image_paths)} images")
        
        logger.info(f"Batch preprocessing complete: {results['success']}/{results['total']} successful")
        
        return results
    
    def save_config(self, output_path: Path) -> None:
        """Save preprocessing configuration."""
        config_dict = asdict(self.config)
        
        # Convert tuples to lists for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Preprocessing config saved to {output_path}")
    
    @classmethod
    def load_config(cls, config_path: Path) -> 'ImagePreprocessor':
        """Load preprocessing configuration from file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to tuples
        tuple_keys = ['target_size', 'clahe_grid_size', 'padding_color',
                      'imagenet_mean', 'imagenet_std']
        for key in tuple_keys:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        
        config = PreprocessingConfig(**config_dict)
        return cls(config)


def main():
    """Example usage."""
    from PIL import Image
    
    # Create preprocessor with default config
    preprocessor = ImagePreprocessor()
    
    # Save config for reproducibility
    config_path = Path("datasets/aptos2019/outputs/preprocessing_config.json")
    preprocessor.save_config(config_path)
    
    print("Preprocessing module ready")
    print(f"Config saved to: {config_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())