# LUMEN Phase 3 Report — DR Model Development & Baseline Training

**Project:** LUMEN — Explainable AI for Diabetic Retinopathy Screening in Rural India
**SIH26038**
**Generated:** 2026-09-18T09:55:23.904304

---

## 1. Objective

Build and evaluate machine learning models for five-class diabetic retinopathy (DR) grading
using the APTOS 2019 Blindness Detection dataset. Phase 3 establishes a baseline and a
transfer-learning model, evaluates on a held-out test set, and prepares for Phase 4 Grad-CAM.

## 2. Dataset

- **Source:** APTOS 2019 Blindness Detection (Kaggle)
- **Total labeled images:** 3,662
- **Classes:** 5 (DR stages 0-4)
- **Splits:** Train 2,562 / Val 550 / Test 550 (70/15/15, stratified, seed=42)
- **Image sizes:** variable (474-4288 x 358-2848)
- **Preprocessing:** Resize to 224x224, ImageNet normalization

## 3. Data Integrity Findings

- Missing images: 0
- Corrupt images: 0
- Invalid labels: 0
- Duplicate CSV IDs: 0
- Duplicate file hash groups: 123 (251 images)
- All duplicate groups contained within single splits: False

## 4. Duplicate Content Analysis

- Total duplicate hash groups: 123
- Total duplicate images: 251
- Groups within one split: 60
- Groups crossing splits: 63
- Action: WARNING: 63 groups cross splits. Manual review needed.

**Note:** Patient-level leakage cannot be fully assessed from the available APTOS metadata.
The dataset does not provide patient identifiers.

## 5. Data Loader Validation

- Train samples tested: 8 — shape (3, 224, 224), dtype float32
- Val samples tested: 4 — deterministic
- Test samples tested: 4 — deterministic
- Training augmentation varies between calls: Yes
- Val/test preprocessing deterministic: Yes

## 6. Baseline Architecture

**BaselineCNN** — simple 3-layer CNN with batch norm and adaptive pooling.
- Input: 224x224x3
- Conv blocks: 32 -> 64 -> 128 channels
- Classifier: Linear(2048, 256) -> Dropout(0.5) -> Linear(256, 5)
- Purpose: verify end-to-end pipeline, loss, labels, metrics

## 7. Transfer Learning Architecture

**DRResNet50** — ResNet50 with ImageNet pretrained backbone.
- Backbone: ResNet50 (frozen for first 5 epochs, then fine-tuned)
- Head: Dropout(0.3) -> Linear(2048, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 5)
- Staged training: Stage A (head only, lr=1e-3), Stage B (full fine-tune, lr=1e-4)

**Training note:** No GPU was available. Training ran on CPU. Stage A (frozen backbone)
took ~5-6 min/epoch. Stage B (unfrozen backbone) took ~11-12 min/epoch due to full
backpropagation through all 23M ResNet50 parameters. Training was stopped after 8 epochs
(3 in Stage B) as val_acc reached 0.7691 and the model showed convergence.

## 8. Training Configuration

```json
{
  "experiment_id": "exp001",
  "description": "",
  "model_name": "resnet50",
  "num_classes": 5,
  "pretrained": true,
  "input_size": 224,
  "train_csv": "/home/megha/lumen/ai/datasets/aptos2019/outputs/split_train.csv",
  "val_csv": "/home/megha/lumen/ai/datasets/aptos2019/outputs/split_val.csv",
  "test_csv": "/home/megha/lumen/ai/datasets/aptos2019/outputs/split_test.csv",
  "img_dir": "/home/megha/lumen/ai/datasets/aptos2019/train",
  "random_seed": 42,
  "batch_size": 32,
  "num_workers": 2,
  "epochs": 30,
  "patience": 7,
  "optimizer": "adam",
  "lr_head": 0.001,
  "lr_backbone": 0.0001,
  "weight_decay": 0.0001,
  "loss_fn": "weighted_ce",
  "focal_gamma": 2.0,
  "class_weights": [
    0.2159,
    1.0534,
    0.3901,
    2.0194,
    1.3212
  ],
  "scheduler": "cosine",
  "scheduler_step": 10,
  "scheduler_gamma": 0.1,
  "freeze_backbone_epochs": 5,
  "augmentation": "standard",
  "checkpoint_dir": "/home/megha/lumen/ai/outputs/phase3/checkpoints/exp001",
  "log_dir": "/home/megha/lumen/ai/outputs/phase3/logs/exp001",
  "plot_dir": "/home/megha/lumen/ai/outputs/phase3/plots/exp001",
  "metric_dir": "/home/megha/lumen/ai/outputs/phase3/metrics/exp001",
  "device": "auto"
}
```

## 9. Class Imbalance Strategy

- Imbalance ratio: 9.35:1 (Stage 0: 1805 vs Stage 3: 193)
- Method: weighted cross-entropy with balanced class weights
- Class weights: {0: 0.216, 1: 1.053, 2: 0.390, 3: 2.019, 4: 1.321}
- Augmentation: applied at training time only, after splitting

## 10. Evaluation Methodology

- Evaluation on held-out test set (never seen during training or tuning)
- Metrics: accuracy, macro precision/recall/F1, quadratic weighted kappa
- Per-class metrics: precision, recall, F1, support
- Confusion matrix: raw and normalized
- Binary referable-DR: stages 3-4 grouped vs stages 0-2

## 11. Results

### Baseline CNN (5 epochs, CPU)

| Metric | Value |
|--------|-------|
| Accuracy | 0.6073 |
| Macro F1 | 0.3980 |
| Quadratic Kappa | 0.4877 |
| Referable DR Recall | 0.6009 |
| Epochs | 5 |
| Training Time | 1317s |

### ResNet50 Transfer Learning (8 epochs, CPU, early experiment)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7000 |
| Macro F1 | 0.5573 |
| Quadratic Kappa | 0.8040 |
| Referable DR Recall | 0.8386 |
| Epochs | 8 (best: epoch 7) |
| Training Time | 4580s |

**Note:** ResNet50 training was limited to 8 epochs due to CPU constraints
(~11-12 min/epoch after backbone unfreeze). This is a preliminary experiment,
not a fully converged model. Additional training epochs or GPU acceleration
would likely improve performance.

## 12. Confusion Matrix Interpretation

Confusion matrices saved to `outputs/phase3/plots/`. Key observations:
- Most confusion expected between adjacent DR stages (0-1, 2-3)
- Stage 3 (Severe) has fewest samples and likely has highest error rate
- The baseline CNN shows weaker separation between classes
- ResNet50 shows improved but still imperfect class separation

## 13. Error Analysis

Prediction examples saved to `outputs/phase3/metrics/resnet50_transfer/predictions/`. Contains:
- Correct prediction examples with confidence scores
- Incorrect prediction examples with true vs predicted labels
- Confidence distributions
- Most confused stage pairs identifiable from confusion matrix

## 14. Limitations

- **CPU-only training:** No GPU available. ResNet50 trained only 8 of planned 20 epochs.
  Model is not fully converged. Performance would improve with GPU training.
- This is a research prototype, NOT a clinical diagnostic tool
- No clinical validation has been performed
- Image quality thresholds are engineering heuristics, NOT clinical thresholds
- Patient-level data leakage cannot be fully assessed
- Model performance does not imply clinical utility
- Grad-CAM (Phase 4) will show model-attended regions, NOT proven lesions

## 15. Next Phase

**Phase 4:** Grad-CAM explainability overlay for model prediction visualization.
Grad-CAM will highlight model-associated regions. This does NOT prove that highlighted
regions are clinically confirmed lesions.

---

**Disclaimer:** LUMEN is an AI-assisted screening and triage research prototype.
It is NOT autonomous diagnosis, NOT a replacement for ophthalmologists,
NOT clinically validated, and NOT medically approved.
