# LUMEN Phase 4 Report — Grad-CAM Explainability

**Project:** LUMEN — Explainable AI for Diabetic Retinopathy Screening in Rural India
**SIH26038**
**Generated:** 2026-09-18T10:04:18.670810

---

## 1. Objective

Implement Grad-CAM (Gradient-weighted Class Activation Mapping) explainability
for the LUMEN DR grading model. Grad-CAM produces heatmaps indicating which
spatial regions of a fundus image most influenced the model's prediction.

## 2. Model Checkpoint Used

- **Model:** DRResNet50 (ResNet50 backbone + custom classification head)
- **Checkpoint:** `outputs/phase3/checkpoints/resnet50_transfer/best_model.pt`
- **Training epoch:** 7
- **Validation loss:** 0.8510
- **Test accuracy:** 0.7000 (from Phase 3 evaluation)
- **Test macro F1:** 0.5573

## 3. Grad-CAM Method

Grad-CAM computes class-discriminative localization maps by:

1. Passing image through model, obtaining prediction
2. Computing gradients of target class score w.r.t. feature maps of last conv layer
3. Pooling gradients globally to get channel importance weights
4. Weighted combination of feature maps + ReLU
5. Normalizing heatmap to [0, 1]

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

## 4. Target Layer

- **Layer:** `backbone.layer4` (ResNet50's last residual block group)
- **Output shape:** (batch, 2048, 7, 7)
- **Justification:** layer4 is the final convolutional feature extractor before
  global average pooling and the classification head. It captures high-level
  semantic features most relevant to the classification.

## 5. Preprocessing

Identical to Phase 3 evaluation preprocessing:

- Resize to 224x224 (bilinear interpolation)
- Convert to tensor (0-1 range)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- NO random augmentation (deterministic)

## 6. Target-Class Methodology

- **Default:** Grad-CAM for the predicted class (most common use case)
- **Optional:** Grad-CAM for any explicitly requested class
- Returns: predicted class, all class probabilities, confidence, target class, heatmap

## 7. Visualization Examples

Generated 15 examples from the held-out test set:
- 10 correct predictions
- 5 incorrect predictions
- Examples from multiple DR stages

Visualizations saved to: `outputs/phase4/visualizations/`
Examples metadata: `outputs/phase4/gradcam_examples.json`

Each visualization is a 3-panel image:
1. Original fundus photograph
2. Grad-CAM heatmap (jet colormap)
3. Original + Grad-CAM overlay (alpha=0.4)

## 8. Technical Validation

- Total images validated: 15
- Passed all checks: 15
- Failed: 0

Checks performed:
- Heatmap dimensions: 7x7 (matches layer4 output)
- Heatmap normalized to [0, 1]
- No NaN values in heatmap
- No Inf values in heatmap
- Image alignment preserved
- Prediction correctness
- Probability sum = 1.0

## 9. Limitations

1. **Grad-CAM is NOT lesion detection.** It shows which regions the model
   attended to, not whether those regions contain clinically confirmed lesions.

2. **Resolution limitation:** The 7x7 heatmap is coarse. Fine-grained
   localization is not possible with standard Grad-CAM on ResNet50.

3. **Model-dependent:** Heatmaps reflect the specific model's learned features,
   which may not align with clinical expectations.

4. **Not clinically validated:** The LUMEN Grad-CAM implementation is an
   explainability aid for the research prototype and has not been
   clinically validated.

5. **CPU-only inference:** All Grad-CAM generation was performed on CPU.
   No GPU acceleration was used.

6. **Training limitation:** The underlying ResNet50 was trained for only 8
   of 20 planned epochs (CPU constraint). Model performance may improve
   with additional training.

## 10. Next Phase

**Phase 5:** LUMEN API and demonstration interface.
- REST API for model inference and Grad-CAM generation
- Integration with Grad-CAM explainability
- Demo interface for SIH presentation

---

**Disclaimer:** LUMEN is an AI-assisted screening and triage research prototype.
Grad-CAM provides model attention visualization, NOT clinical diagnosis.
Highlighted regions indicate model-relevant features, NOT confirmed lesions.
