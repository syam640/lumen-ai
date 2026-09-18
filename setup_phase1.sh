#!/bin/bash
# LUMEN Phase 1 Setup Script
# Sets up environment and downloads APTOS 2019 dataset

set -e  # Exit on error

echo ""
echo "=========================================="
echo "LUMEN Phase 1: Dataset Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: Please run this script from lumen/ai/ directory"
    echo "Usage: cd lumen/ai && ./setup_phase1.sh"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${CYAN}[1/4]${NC} Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "  ${RED}✗${NC} Python 3 not found"
    echo "    Please install Python 3.8+"
    exit 1
fi

# Step 2: Install requirements
echo ""
echo -e "${CYAN}[2/4]${NC} Installing requirements..."
pip install -q -r requirements.txt
echo -e "  ${GREEN}✓${NC} Requirements installed"

# Step 3: Check Kaggle credentials
echo ""
echo -e "${CYAN}[3/4]${NC} Checking Kaggle credentials..."
if [ -f ~/.kaggle/kaggle.json ] || [ -n "$KAGGLE_KEY" ]; then
    echo -e "  ${GREEN}✓${NC} Kaggle credentials found"
else
    echo -e "  ${YELLOW}⚠${NC} Kaggle credentials not configured"
    echo ""
    echo "  To set up Kaggle API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Download kaggle.json"
    echo -e "  4. Place at: ${CYAN}~/.kaggle/kaggle.json${NC}"
    echo -e "  5. Run: ${CYAN}chmod 600 ~/.kaggle/kaggle.json${NC}"
    echo ""
    echo "  Or set environment variables:"
    echo -e "    ${CYAN}export KAGGLE_USERNAME=your_username${NC}"
    echo -e "    ${CYAN}export KAGGLE_KEY=your_key${NC}"
fi

# Step 4: Download dataset
echo ""
echo -e "${CYAN}[4/4]${NC} Downloading APTOS 2019 dataset..."
if [ -d "datasets/aptos2019/train" ] && [ "$(ls -A datasets/aptos2019/train/*.jpg 2>/dev/null | wc -l)" -gt "100" ]; then
    echo -e "  ${GREEN}✓${NC} Dataset already exists"
else
    echo "  Running download script..."
    python3 download_dataset.py
fi

# Verify
echo ""
echo "=========================================="
echo "Verifying setup..."
echo "=========================================="
python3 verify_setup.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run dataset inspection:"
echo -e "     ${CYAN}python datasets/inspect_dataset.py${NC}"
echo ""
echo "  2. Review outputs in:"
echo -e "     ${CYAN}datasets/aptos2019/outputs/${NC}"
echo ""