#!/usr/bin/env python3
"""
LUMEN — APTOS 2019 Automatic Dataset Downloader

Downloads the APTOS 2019 Blindness Detection dataset from Kaggle
and organizes it into the project structure.

Usage:
    python download_dataset.py
    python download_dataset.py --force  # Re-download even if dataset exists
    
Environment Variables:
    LUMEN_DOWNLOAD_TIMEOUT: Timeout in seconds for Kaggle download (default: no limit)
    LUMEN_DOWNLOAD_RETRIES: Number of retry attempts (default: 3)
    LUMEN_RETRY_DELAY: Delay between retries in seconds (default: 30)
    
Prerequisites:
    1. Install Kaggle CLI: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to https://www.kaggle.com/settings
       - Create API token (downloads kaggle.json)
       - Place at ~/.kaggle/kaggle.json (Linux/Mac)
         or C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)
    3. Accept competition rules at:
       https://www.kaggle.com/c/aptos2019-blindness-detection
"""

import os
import sys
import shutil
import logging
import subprocess
import zipfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

# Add datasets directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "datasets"))
from dataset_manager import DatasetManager, KAGGLE_COMPETITION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration from environment
DOWNLOAD_TIMEOUT = int(os.environ.get('LUMEN_DOWNLOAD_TIMEOUT', '0'))  # 0 = no timeout
DOWNLOAD_RETRIES = int(os.environ.get('LUMEN_DOWNLOAD_RETRIES', '3'))
RETRY_DELAY = int(os.environ.get('LUMEN_RETRY_DELAY', '30'))


# Colors for terminal output
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
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.END}\n")


def print_step(step: int, total: int, text: str, status: str = "IN PROGRESS") -> None:
    """Print formatted step."""
    status_color = Colors.YELLOW if status == "IN PROGRESS" else (
        Colors.GREEN if status == "PASS" else Colors.RED
    )
    print(f"  [{step}/{total}] {text:<30} {status_color}{status}{Colors.END}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"\n  {Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"\n  {Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"\n  {Colors.YELLOW}⚠ {text}{Colors.END}")


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def validate_zip_integrity(zip_path: Path) -> Tuple[bool, str]:
    """
    Validate ZIP archive integrity.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not zip_path.exists():
        return False, f"File does not exist: {zip_path}"
    
    file_size = zip_path.stat().st_size
    if file_size == 0:
        return False, f"File is empty: {zip_path}"
    
    # Check for ZIP magic number
    try:
        with open(zip_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'PK\x03\x04':
                return False, f"Not a valid ZIP file (invalid magic number): {zip_path}"
    except Exception as e:
        return False, f"Failed to read file header: {e}"
    
    # Try to open and test the ZIP
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test all archives - this verifies the central directory
            bad_file = zip_ref.testzip()
            if bad_file is not None:
                return False, f"Corrupt file in archive: {bad_file}"
            
            # Check if archive has any files
            if len(zip_ref.namelist()) == 0:
                return False, f"ZIP archive is empty (no files)"
            
            return True, ""
            
    except zipfile.BadZipFile as e:
        return False, f"Invalid ZIP file (bad central directory): {e}"
    except Exception as e:
        return False, f"Failed to validate ZIP: {e}"


class KaggleDownloader:
    """Handles Kaggle dataset download."""
    
    def __init__(self):
        self.manager = DatasetManager()
        self.kaggle_available = False
        self.credentials_available = False
        
    def check_kaggle_installed(self) -> bool:
        """Check if Kaggle CLI is installed."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Kaggle CLI found: {version}")
                return True
        except FileNotFoundError:
            logger.debug("Kaggle CLI not found in PATH")
        except subprocess.TimeoutExpired:
            logger.warning("Kaggle CLI check timed out")
        except Exception as e:
            logger.debug(f"Kaggle check failed: {e}")
        
        return False
    
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle API credentials are configured."""
        # Check environment variables first
        if os.environ.get('KAGGLE_KEY') and os.environ.get('KAGGLE_USERNAME'):
            logger.info("Kaggle credentials found in environment variables")
            return True
        
        # Check kaggle.json file
        kaggle_json_paths = [
            Path.home() / ".kaggle" / "kaggle.json",
            Path(os.environ.get('APPDATA', '')) / "kaggle" / "kaggle.json" if os.name == 'nt' else None,
        ]
        
        for path in kaggle_json_paths:
            if path and path.exists():
                # Set permissions on Unix-like systems
                if os.name != 'nt':
                    try:
                        os.chmod(path, 0o600)
                    except Exception:
                        pass
                
                logger.info(f"Kaggle credentials found at: {path}")
                return True
        
        logger.debug("No Kaggle credentials found")
        return False
    
    def check_disk_space(self, required_gb: float = 8.0) -> bool:
        """Check if sufficient disk space is available."""
        try:
            stat = shutil.disk_usage(str(self.manager.aptos_dir))
            free_gb = stat.free / (1024 ** 3)
            
            if free_gb >= required_gb:
                logger.info(f"Disk space available: {free_gb:.1f} GB")
                return True
            else:
                logger.warning(f"Insufficient disk space: {free_gb:.1f} GB available, {required_gb:.1f} GB required")
                return False
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume OK if check fails
    
    def cleanup_incomplete_downloads(self) -> None:
        """Remove any incomplete/corrupt ZIP files from raw directory."""
        raw_dir = self.manager.raw_dir
        
        for zip_path in raw_dir.glob("*.zip"):
            is_valid, error_msg = validate_zip_integrity(zip_path)
            if not is_valid:
                logger.warning(f"Removing invalid ZIP: {zip_path.name} - {error_msg}")
                try:
                    # Rename instead of delete for safety
                    backup_name = zip_path.with_suffix('.zip.corrupt')
                    if backup_name.exists():
                        backup_name.unlink()
                    zip_path.rename(backup_name)
                    logger.info(f"Renamed to: {backup_name.name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {zip_path.name}: {e}")
    
    def download_competition_data(self) -> Tuple[bool, Optional[Path]]:
        """
        Download dataset from Kaggle competition with retry logic.
        
        Returns:
            Tuple of (success, zip_path_or_none)
        """
        raw_dir = self.manager.raw_dir
        
        # Clean up any previous incomplete downloads
        self.cleanup_incomplete_downloads()
        
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            logger.info(f"Download attempt {attempt}/{DOWNLOAD_RETRIES}")
            
            start_time = time.time()
            
            try:
                # Build command
                cmd = [
                    "kaggle", "competitions", "download",
                    "-c", KAGGLE_COMPETITION,
                    "-p", str(raw_dir)
                ]
                
                # Set timeout if configured
                timeout_value = DOWNLOAD_TIMEOUT if DOWNLOAD_TIMEOUT > 0 else None
                
                logger.info(f"Running: {' '.join(cmd)}")
                if timeout_value:
                    logger.info(f"Timeout: {timeout_value}s")
                else:
                    logger.info("Timeout: none (will run until complete)")
                
                # Run download
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_value
                )
                
                elapsed = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    
                    # Handle specific error codes
                    if "403" in error_msg or "Forbidden" in error_msg:
                        logger.error("Access denied. Please accept competition rules at:")
                        logger.error(f"https://www.kaggle.com/c/{KAGGLE_COMPETITION}")
                        return False, None
                    elif "401" in error_msg or "Unauthorized" in error_msg:
                        logger.error("Authentication failed. Please check your Kaggle credentials.")
                        return False, None
                    elif "timeout" in error_msg.lower():
                        logger.warning(f"Download timed out after {format_elapsed(elapsed)}")
                        if attempt < DOWNLOAD_RETRIES:
                            logger.info(f"Retrying in {RETRY_DELAY}s...")
                            time.sleep(RETRY_DELAY)
                            continue
                        else:
                            logger.error("Max retries reached")
                            return False, None
                    else:
                        logger.error(f"Download failed (attempt {attempt}): {error_msg}")
                        if attempt < DOWNLOAD_RETRIES:
                            logger.info(f"Retrying in {RETRY_DELAY}s...")
                            time.sleep(RETRY_DELAY)
                            continue
                        else:
                            return False, None
                
                # Download succeeded - find the ZIP file
                logger.info(f"Download completed in {format_elapsed(elapsed)}")
                
                # Look for downloaded ZIP files
                zip_files = list(raw_dir.glob("*.zip"))
                
                if not zip_files:
                    logger.error("No ZIP files found after download")
                    if attempt < DOWNLOAD_RETRIES:
                        logger.info(f"Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        return False, None
                
                # Find the most likely competition ZIP
                # APTOS typically downloads as aptos2019-blindness-detection.zip
                target_zip = None
                for zip_path in zip_files:
                    if "aptos" in zip_path.name.lower() or "blindness" in zip_path.name.lower():
                        target_zip = zip_path
                        break
                
                if target_zip is None:
                    # Use the most recent ZIP
                    target_zip = max(zip_files, key=lambda p: p.stat().st_mtime)
                
                # Validate the ZIP
                logger.info(f"Validating: {target_zip.name}")
                is_valid, error_msg = validate_zip_integrity(target_zip)
                
                if not is_valid:
                    logger.warning(f"ZIP validation failed: {error_msg}")
                    # Rename invalid ZIP
                    backup_name = target_zip.with_suffix('.zip.corrupt')
                    if backup_name.exists():
                        backup_name.unlink()
                    target_zip.rename(backup_name)
                    logger.info(f"Renamed invalid ZIP to: {backup_name.name}")
                    
                    if attempt < DOWNLOAD_RETRIES:
                        logger.info(f"Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        logger.error("Max retries reached with invalid ZIP")
                        return False, None
                
                # ZIP is valid
                file_size_mb = target_zip.stat().st_size / (1024 * 1024)
                logger.info(f"ZIP validated: {target_zip.name} ({file_size_mb:.1f} MB)")
                
                return True, target_zip
                
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                logger.warning(f"Download timed out after {format_elapsed(elapsed)}")
                
                # Clean up incomplete download
                for zip_path in raw_dir.glob("*.zip"):
                    is_valid, _ = validate_zip_integrity(zip_path)
                    if not is_valid:
                        backup_name = zip_path.with_suffix('.zip.corrupt')
                        if backup_name.exists():
                            backup_name.unlink()
                        zip_path.rename(backup_name)
                        logger.info(f"Renamed incomplete download to: {backup_name.name}")
                
                if attempt < DOWNLOAD_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    logger.error("Max retries reached after timeout")
                    return False, None
                    
            except Exception as e:
                logger.error(f"Download failed with exception (attempt {attempt}): {e}")
                if attempt < DOWNLOAD_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return False, None
        
        return False, None
    
    def extract_files(self, zip_path: Path) -> bool:
        """Extract downloaded archive with validation."""
        if not zip_path.exists():
            logger.error(f"ZIP file not found: {zip_path}")
            return False
        
        # Final validation before extraction
        logger.info(f"Final validation of {zip_path.name}...")
        is_valid, error_msg = validate_zip_integrity(zip_path)
        if not is_valid:
            logger.error(f"Cannot extract invalid ZIP: {error_msg}")
            return False
        
        raw_dir = self.manager.raw_dir
        
        logger.info(f"Extracting {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List contents before extraction
                file_list = zip_ref.namelist()
                logger.info(f"Archive contains {len(file_list)} files")
                
                # Extract
                zip_ref.extractall(raw_dir)
                logger.info(f"Extraction complete to {raw_dir}")
            
            # Remove zip after successful extraction
            try:
                zip_path.unlink()
                logger.debug(f"Removed {zip_path.name}")
            except Exception as e:
                logger.warning(f"Could not remove ZIP: {e}")
            
            return True
            
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract (bad ZIP): {e}")
            # Rename the corrupt ZIP
            backup_name = zip_path.with_suffix('.zip.corrupt')
            if backup_name.exists():
                backup_name.unlink()
            zip_path.rename(backup_name)
            logger.info(f"Renamed corrupt ZIP to: {backup_name.name}")
            return False
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def organize_files(self) -> bool:
        """Organize extracted files into canonical structure."""
        raw_dir = self.manager.raw_dir
        
        # Find train.csv
        csv_files = list(raw_dir.rglob("train.csv"))
        if not csv_files:
            logger.error("train.csv not found after extraction")
            return False
        
        # Move train.csv to metadata directory
        train_csv = csv_files[0]
        target_csv = self.manager.get_train_csv_path()
        
        try:
            shutil.copy2(train_csv, target_csv)
            logger.info(f"Copied train.csv to {target_csv}")
        except Exception as e:
            logger.error(f"Failed to copy train.csv: {e}")
            return False
        
        # Find and move training images
        # Look for train_images directory or similar
        train_img_dirs = [
            raw_dir / "train_images",
            raw_dir / "train" / "train_images",
            raw_dir,
        ]
        
        source_dir = None
        for dir_path in train_img_dirs:
            if dir_path.exists():
                # Check if it contains images (JPG, JPEG, or PNG)
                images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg")) + list(dir_path.glob("*.png"))
                if len(images) > 100:  # Likely the training images
                    source_dir = dir_path
                    break
        
        if source_dir is None:
            # Try to find any directory with many images
            for dir_path in raw_dir.rglob("*"):
                if dir_path.is_dir():
                    images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg")) + list(dir_path.glob("*.png"))
                    if len(images) > 100:
                        source_dir = dir_path
                        break
        
        if source_dir is None:
            logger.error("Could not locate training images directory")
            return False
        
        # Copy images to train directory
        target_dir = self.manager.train_dir
        logger.info(f"Copying images from {source_dir} to {target_dir}...")
        
        try:
            images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg")) + list(source_dir.glob("*.png"))
            for idx, img_path in enumerate(images):
                shutil.copy2(img_path, target_dir / img_path.name)
                if (idx + 1) % 500 == 0:
                    logger.info(f"  Copied {idx + 1}/{len(images)} images")
            
            logger.info(f"Copied {len(images)} images")
        except Exception as e:
            logger.error(f"Failed to copy images: {e}")
            return False
        
        return True
    
    def run(self, force: bool = False) -> bool:
        """Run the complete download pipeline."""
        total_steps = 7
        
        print_header("LUMEN — APTOS DATASET SETUP")
        
        # Step 1: Check Python
        print_step(1, total_steps, "Checking Python")
        print_step(1, total_steps, "Checking Python", "PASS")
        
        # Step 2: Check Kaggle CLI
        print_step(2, total_steps, "Checking Kaggle CLI", "IN PROGRESS")
        self.kaggle_available = self.check_kaggle_installed()
        if not self.kaggle_available:
            print_step(2, total_steps, "Checking Kaggle CLI", "FAIL")
            print_error("Kaggle CLI is not installed.")
            print("\n  Install it with:")
            print(f"    {Colors.CYAN}pip install kaggle{Colors.END}")
            return False
        print_step(2, total_steps, "Checking Kaggle CLI", "PASS")
        
        # Step 3: Check credentials
        print_step(3, total_steps, "Checking credentials", "IN PROGRESS")
        self.credentials_available = self.check_kaggle_credentials()
        if not self.credentials_available:
            print_step(3, total_steps, "Checking credentials", "FAIL")
            print_error("Kaggle credentials not configured.")
            print("\n  To set up Kaggle API credentials:")
            print("  1. Go to https://www.kaggle.com/settings")
            print("  2. Click 'Create New API Token'")
            print("  3. Download kaggle.json")
            print(f"  4. Place at: {Colors.CYAN}~/.kaggle/kaggle.json{Colors.END}")
            print(f"\n  Then run: {Colors.CYAN}chmod 600 ~/.kaggle/kaggle.json{Colors.END}")
            return False
        print_step(3, total_steps, "Checking credentials", "PASS")
        
        # Step 4: Check if dataset exists
        print_step(4, total_steps, "Checking dataset", "IN PROGRESS")
        if self.manager.is_dataset_complete() and not force:
            print_step(4, total_steps, "Checking dataset", "PASS")
            print_success("APTOS dataset already exists and passed verification.")
            
            # Show verification
            report = self.manager.verify_dataset()
            self._print_dataset_summary(report)
            return True
        
        if force:
            print_warning("Force mode: Re-downloading dataset")
        print_step(4, total_steps, "Checking dataset", "NOT FOUND")
        
        # Step 5: Check disk space
        print_step(5, total_steps, "Checking disk space", "IN PROGRESS")
        if not self.check_disk_space():
            print_step(5, total_steps, "Checking disk space", "FAIL")
            return False
        print_step(5, total_steps, "Checking disk space", "PASS")
        
        # Step 6: Download with retry
        print_step(6, total_steps, "Downloading from Kaggle", "IN PROGRESS")
        download_start = time.time()
        
        success, zip_path = self.download_competition_data()
        
        download_elapsed = time.time() - download_start
        
        if not success:
            print_step(6, total_steps, "Downloading from Kaggle", "FAIL")
            print_error(f"Download failed after {format_elapsed(download_elapsed)}")
            return False
        
        print_step(6, total_steps, "Downloading from Kaggle", "PASS")
        logger.info(f"Total download time: {format_elapsed(download_elapsed)}")
        
        # Step 7: Extract and organize
        print_step(7, total_steps, "Extracting and organizing", "IN PROGRESS")
        
        if not self.extract_files(zip_path):
            print_step(7, total_steps, "Extracting and organizing", "FAIL")
            return False
        
        if not self.organize_files():
            print_step(7, total_steps, "Extracting and organizing", "FAIL")
            return False
        print_step(7, total_steps, "Extracting and organizing", "PASS")
        
        # Verify and generate report
        print_header("VERIFICATION")
        
        report = self.manager.verify_dataset()
        self.manager.generate_manifest()
        
        self._print_dataset_summary(report)
        
        if report['status'] == 'PASS':
            print_success("Dataset download and verification complete!")
            return True
        else:
            print_error(f"Dataset verification {report['status']}")
            return False
    
    def _print_dataset_summary(self, report: dict) -> None:
        """Print formatted dataset summary."""
        print(f"\n  Dataset: APTOS 2019 Blindness Detection")
        print(f"  Status: {report['status']}")
        print(f"\n  CSV Records: {report['csv_records']}")
        print(f"  Images Found: {report['images_found']}")
        
        if report.get('label_distribution'):
            print(f"\n  Class Distribution:")
            for stage_name, stats in report['label_distribution'].items():
                print(f"    {stage_name}: {stats['count']} ({stats['percentage']}%)")
        
        if report['missing_images'] > 0:
            print_warning(f"Missing images: {report['missing_images']}")
        
        if report['corrupt_images'] > 0:
            print_warning(f"Corrupt images: {report['corrupt_images']}")
        
        # Disk usage
        usage = self.manager.get_disk_usage()
        print(f"\n  Disk Usage: {usage['total_size_mb']:.1f} MB ({usage['file_count']} files)")
        
        print(f"\n  Outputs:")
        print(f"    {self.manager.outputs_dir}")


def main():
    """Main entry point."""
    downloader = KaggleDownloader()
    
    # Check for force flag
    force = "--force" in sys.argv or "-f" in sys.argv
    
    success = downloader.run(force=force)
    
    if success:
        print_header("NEXT STEPS")
        print("  1. Run dataset inspection:")
        print(f"     {Colors.CYAN}python datasets/inspect_dataset.py{Colors.END}")
        print("\n  2. Review outputs in:")
        print(f"     {Colors.CYAN}datasets/aptos2019/outputs/{Colors.END}")
    else:
        print_header("TROUBLESHOOTING")
        print("  If you're having issues:")
        print("  1. Verify Kaggle credentials: kaggle datasets list")
        print("  2. Accept competition rules at:")
        print(f"     https://www.kaggle.com/c/{KAGGLE_COMPETITION}")
        print("  3. Check internet connection")
        print(f"  4. See: {Colors.CYAN}datasets/DATASET_INSTRUCTIONS.md{Colors.END}")
        print(f"\n  Configuration:")
        print(f"    Timeout: {DOWNLOAD_TIMEOUT if DOWNLOAD_TIMEOUT > 0 else 'none'}")
        print(f"    Retries: {DOWNLOAD_RETRIES}")
        print(f"    Retry delay: {RETRY_DELAY}s")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())