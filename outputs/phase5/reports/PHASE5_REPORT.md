# LUMEN Phase 5 Report — API + Demonstration Interface

**Project:** LUMEN — Explainable AI for Diabetic Retinopathy Screening in Rural India
**SIH26038**
**Generated:** 2026-09-18

---

## 1. Architecture

```
ai/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   ├── inference.py      # Inference service (model + Grad-CAM)
│   │   ├── quality.py        # Image quality checking
│   │   └── schemas.py        # Pydantic response models
│   ├── triage/
│   │   └── rules.py          # Rule-based triage engine
│   ├── training/
│   │   ├── models.py         # DRResNet50 architecture
│   │   ├── dataset.py        # Preprocessing transforms
│   │   └── evaluate.py       # Evaluation utilities
│   └── explainability/
│       ├── gradcam.py        # Grad-CAM implementation
│       └── visualization.py  # 3-panel visualization
├── frontend/                  # React + TypeScript + Vite
│   └── src/App.tsx           # Single-page screening UI
├── demo.py                   # CLI fallback for SIH demo
├── run_gradcam.py            # Grad-CAM batch generation
└── tests/
    └── test_api.py           # 22 tests (all passing)
```

## 2. API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, model status |
| `/api/v1/screen` | POST | Full screening pipeline |
| `/api/v1/quality` | POST | Quality check only |

### POST /api/v1/screen

**Request:** Multipart form with `file` (fundus image)

**Response:**
```json
{
  "success": true,
  "prediction": {"stage": 2, "label": "Moderate", "confidence": 0.51},
  "probabilities": {"0": 0.0001, "1": 0.096, "2": 0.510, "3": 0.214, "4": 0.179},
  "quality": {"status": "POOR", "brightness": 51.39, "contrast": 31.3, "sharpness": 6.26, ...},
  "triage": {"priority": "MODERATE", "reason": "..."},
  "explainability": {"method": "Grad-CAM", "heatmap_available": true, "visualization_url": "/visualizations/..."},
  "inference_time_ms": 12494
}
```

## 3. Inference Pipeline

1. **Upload** → validate file type, size, readability
2. **Quality check** → brightness, contrast, sharpness (engineering heuristics)
3. **Preprocessing** → resize 224x224, ImageNet normalization
4. **Model inference** → ResNet50 (epoch 7 checkpoint, val_loss=0.8510)
5. **Probability distribution** → softmax over 5 DR stages
6. **Grad-CAM** → heatmap for predicted class
7. **Triage** → rule-based priority assignment
8. **Response** → structured JSON

Model loaded once at startup. Single-image inference ~12.5s (CPU).

## 4. Quality Checking

Reuses Phase 2 quality analysis concepts:
- Sharpness (Laplacian variance)
- Brightness (mean pixel intensity)
- Contrast (std of grayscale)

**Thresholds are engineering heuristics, NOT clinically validated.**

If quality = BAD: inference is skipped with a user-friendly message.

## 5. Model Integration

- **Architecture:** DRResNet50 (ResNet50 backbone + custom head)
- **Checkpoint:** epoch 7, val_loss=0.8510, test_accuracy=0.7000
- **Target layer:** backbone.layer4 (2048 channels, 7x7)
- **Loaded once** at API startup (not per-request)

## 6. Grad-CAM Integration

- Reuses existing `src/explainability/gradcam.py`
- Generates heatmap for predicted class
- Saves 3-panel visualization (original / heatmap / overlay)
- Returns URL for frontend display

## 7. Triage Rules

| Stage | Priority | Action |
|-------|----------|--------|
| 0-1 | LOW | Routine follow-up |
| 2-3 | MODERATE/HIGH | Priority/urgent referral |
| 4 | URGENT | Immediate referral |

**Demonstration rules only.** Not clinically validated referral protocol.
Low-confidence predictions (<50%) include a manual review note.

## 8. Frontend

React + TypeScript + Vite single-page application:
- File upload (drag/drop + picker)
- Image quality display
- DR stage prediction with confidence bars
- Grad-CAM visualization
- Triage priority badge
- Research prototype disclaimer

Start: `cd frontend && npm run dev` (port 5173, proxies to API on 8000)

## 9. Testing

**22/22 tests passing:**
- 6 triage tests (stages 0-4, invalid, low-confidence)
- 3 quality tests (good image, dark image, usability check)
- 8 inference tests (load, prediction, probabilities, Grad-CAM, timing)
- 5 API tests (health, valid image, invalid file, empty file, quality endpoint)

## 10. Performance

- Model load: ~2.2s (once at startup)
- Quality check: ~80ms
- Inference: ~274ms
- Grad-CAM: ~12.1s
- Total screening: ~12.5s

CPU-only. Single-image processing. No batching.

## 11. Local Startup

```bash
# Start API
cd ai
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start frontend (separate terminal)
cd ai/frontend
npm install
npm run dev

# CLI fallback
python3 demo.py --image <fundus_image>
```

## 12. Limitations

1. **Research prototype** — not autonomous diagnosis
2. **CPU-only** — ~12.5s per screening (no GPU)
3. **8/20 epochs trained** — model not fully converged
4. **Quality thresholds** — engineering heuristics, not clinical standards
5. **Triage rules** — demonstration only, require ophthalmologist validation
6. **Grad-CAM** — model attention, not confirmed lesion detection
7. **No clinical validation** — no regulatory approval claimed

## 13. Phase 5 Verification Status

**PHASE 5 STATUS:**
| Check | Status |
|-------|--------|
| Backend starts | PASS (PID 2562, port 8000) |
| Frontend starts | PASS (PID 2655, port 5173) |
| API integration (proxy) | PASS (/health, /api/v1/screen, /api/v1/quality all reachable through frontend proxy) |
| Real image test (APTOS) | PASS (000c1434d8d7.png → Stage 2 Moderate, 51.02%) |
| Grad-CAM generation | PASS (heatmap saved, URL served at /visualizations/) |
| Triage result | PASS (MODERATE priority, correct reason) |
| Production build | PASS (dist/ built: 149.61 KB JS, 0.41 KB HTML) |
| Unit tests | PASS (22/22) |
| Frontend URL | http://localhost:5173 |
| Backend URL | http://localhost:8000 |
| Production URL | http://localhost:8000/app |

**Unresolved issues:** None.

**Verified on:** 2026-09-18

---

**Disclaimer:** LUMEN is an AI-assisted screening and triage research prototype.
It does not provide a definitive diagnosis and does not replace evaluation
by a qualified eye-care professional.
