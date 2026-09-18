#!/usr/bin/env python3
"""
LUMEN — Environment Verification Script

Verifies that the environment is properly configured for dataset acquisition.
"""

import sys
import os
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{'=' * 50}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 50}{Colors.END}\n")


def check_python_version() -> bool:
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  {Colors.RED}✗ Python {version.major}.{version.minor} detected{Colors.END}")
        print(f"    Required: Python 3.8+")
        return False
    print(f"  {Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro}{Colors.END}")
    return True


def check_packages() -> bool:
    """Check required packages."""
    required = ['pandas', 'numpy', 'matplotlib', 'cv2', 'kaggle']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'kaggle':
                # Check if kaggle CLI is available
                result = subprocess.run(
                    ["kaggle", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    raise ImportError("Kaggle CLI not working")
            else:
                __import__(package)
            print(f"  {Colors.GREEN}✓ {package}{Colors.END}")
        except ImportError:
            print(f"  {Colors.RED}✗ {package} not installed{Colors.END}")
            missing.append(package)
        except Exception:
            print(f"  {Colors.RED}✗ {package} not working properly{Colors.END}")
            missing.append(package)
    
    return len(missing) == 0


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured."""
    # Check environment variables
    if os.environ.get('KAGGLE_KEY') and os.environ.get('KAGGLE_USERNAME'):
        print(f"  {Colors.GREEN}✓ Kaggle credentials (environment variables){Colors.END}")
        return True
    
    # Check kaggle.json file
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    
    if os.name == 'nt':  # Windows
        appdata = os.environ.get('APPDATA', '')
        if appdata:
            kaggle_json_paths.append(Path(appdata) / "kaggle" / "kaggle.json")
    
    for path in kaggle_json_paths:
        if path.exists():
            print(f"  {Colors.GREEN}✓ Kaggle credentials ({path}){Colors.END}")
            return True
    
    print(f"  {Colors.YELLOW}⚠ Kaggle credentials not found{Colors.END}")
    return False


def check_dataset_structure() -> bool:
    """Check if dataset structure exists."""
    base_path = Path("datasets/aptos2019")
    
    checks = [
        (base_path / "train", "Train directory"),
        (base_path / "metadata", "Metadata directory"),
        (base_path / "metadata" / "train.csv", "train.csv"),
    ]
    
    all_ok = True
    for path, name in checks:
        if path.exists():
            print(f"  {Colors.GREEN}✓ {name}{Colors.END}")
        else:
            print(f"  {Colors.RED}✗ {name} not found{Colors.END}")
            all_ok = False
    
    # Check image count
    train_path = base_path / "train"
    if train_path.exists():
        # Count both JPG and PNG images
        jpg_count = len(list(train_path.glob("*.jpg")))
        png_count = len(list(train_path.glob("*.png")))
        image_count = jpg_count + png_count
        
        if image_count > 0:
            print(f"  {Colors.GREEN}✓ {image_count} images found in train/ ({jpg_count} JPG, {png_count} PNG){Colors.END}")
        else:
            print(f"  {Colors.YELLOW}⚠ No images found in train/{Colors.END}")
            all_ok = False
    
    return all_ok


def check_competition_access() -> bool:
    """Check if competition rules are accepted."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list", "-s", "aptos2019"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "aptos2019" in result.stdout.lower():
            print(f"  {Colors.GREEN}✓ APTOS competition accessible{Colors.END}")
            return True
        else:
            print(f"  {Colors.YELLOW}⚠ Could not verify competition access{Colors.END}")
            return False
    except Exception:
        print(f"  {Colors.YELLOW}⚠ Could not check competition access{Colors.END}")
        return False


def main():
    """Main verification pipeline."""
    print_header("LUMEN Environment Verification")
    
    results = {}
    
    # Check Python
    print("Checking Python version:")
    results['python'] = check_python_version()
    print()
    
    # Check packages
    print("Checking required packages:")
    results['packages'] = check_packages()
    print()
    
    # Check Kaggle credentials
    print("Checking Kaggle credentials:")
    results['credentials'] = check_kaggle_credentials()
    print()
    
    # Check dataset
    print("Checking dataset structure:")
    results['dataset'] = check_dataset_structure()
    print()
    
    # Check competition access (only if credentials exist)
    if results['credentials']:
        print("Checking competition access:")
        results['competition'] = check_competition_access()
        print()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    all_pass = all(results.values())
    
    if all_pass:
        print(f"  {Colors.GREEN}✓ All checks passed!{Colors.END}")
        print()
        
        if results['dataset']:
            print("Ready to run dataset inspection:")
            print(f"  {Colors.CYAN}python datasets/inspect_dataset.py{Colors.END}")
        else:
            print("Ready to download dataset:")
            print(f"  {Colors.CYAN}python download_dataset.py{Colors.END}")
    else:
        print(f"  {Colors.RED}✗ Some checks failed{Colors.END}")
        print()
        
        if not results['packages']:
            print(f"  Install packages: {Colors.CYAN}pip install -r requirements.txt{Colors.END}")
        
        if not results['credentials']:
            print(f"\n  To configure Kaggle credentials:")
            print(f"  1. Go to https://www.kaggle.com/settings")
            print(f"  2. Click 'Create New API Token'")
            print(f"  3. Place kaggle.json at: {Colors.CYAN}~/.kaggle/kaggle.json{Colors.END}")
            print(f"  4. Run: {Colors.CYAN}chmod 600 ~/.kaggle/kaggle.json{Colors.END}")
        
        if not results['dataset']:
            print(f"\n  To download dataset:")
            print(f"  {Colors.CYAN}python download_dataset.py{Colors.END}")
    
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())