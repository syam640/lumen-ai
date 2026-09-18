# LUMEN AI Module

Explainable AI for Diabetic Retinopathy Screening in Rural India

## Quick Start

### 1. Setup Environment

```bash
cd lumen/ai
pip install -r requirements.txt
```

### 2. Configure Kaggle Authentication

**One-time setup:**

1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it at:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`

5. Set permissions (Linux/Mac):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

6. Accept competition rules:
   https://www.kaggle.com/c/aptos2019-blindness-detection

### 3. Download Dataset

```bash
python download_dataset.py
```

**Note:** The APTOS dataset is ~10GB. Download may take 15-30 minutes depending on your internet connection. The downloader will:
- Run until complete (no timeout)
- Retry automatically on network failures (3 attempts by default)
- Validate ZIP integrity before extraction
- Handle corrupt downloads safely

### 4. Verify Setup

```bash
python verify_setup.py
```

### 5. Run Dataset Inspection

```bash
python datasets/inspect_dataset.py
```

## Project Structure

```
lumen/ai/
├── datasets/
│   ├── dataset_manager.py      # Dataset utilities
│   ├── inspect_dataset.py      # Dataset inspection
│   ├── DATASET_INSTRUCTIONS.md # Download guide
│   └── aptos2019/              # Dataset directory
│       ├── train/              # Training images (PNG)
│       ├── test/               # Test images
│       ├── metadata/           # Labels and manifest
│       ├── raw/                # Downloaded archives
│       └── outputs/            # Generated reports
├── download_dataset.py         # Automatic downloader
├── verify_setup.py            # Environment verification
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Dataset

**APTOS 2019 Blindness Detection**

- Source: Kaggle competition
- Total Images: 3,662 fundus photographs (training set)
- Labels: DR severity grades 0-4
- Classes: No DR, Mild, Moderate, Severe, Proliferative DR
- Format: PNG images

## Commands

| Command | Description |
|---------|-------------|
| `python download_dataset.py` | Download APTOS dataset |
| `python download_dataset.py --force` | Force re-download |
| `python verify_setup.py` | Verify environment |
| `python datasets/inspect_dataset.py` | Generate dataset report |

## Download Configuration

Environment variables for download behavior:

```bash
# Timeout in seconds (0 = no timeout, default)
export LUMEN_DOWNLOAD_TIMEOUT=0

# Number of retry attempts (default: 3)
export LUMEN_DOWNLOAD_RETRIES=3

# Delay between retries in seconds (default: 30)
export LUMEN_RETRY_DELAY=30
```

## Outputs

After running inspection, find reports in:
```
datasets/aptos2019/outputs/
├── class_distribution.png
├── sample_images.png
├── dimension_distribution.png
├── dataset_report.json
└── DATASET_REPORT.md
```

## Troubleshooting

**Authentication Error:**
- Verify credentials: `kaggle datasets list`
- Accept competition rules at Kaggle website

**Download Error:**
- Check internet connection
- Verify disk space (10 GB+ required)
- Check environment variables for timeout/retry settings

**Corrupt ZIP Error:**
- The downloader automatically handles corrupt files
- Run `python download_dataset.py --force` to retry

**Missing Images:**
- Re-run: `python download_dataset.py --force`

## Next Steps

After Phase 1 (dataset acquisition):
1. Review dataset report
2. Analyze class imbalance
3. Proceed to Phase 2: Preprocessing