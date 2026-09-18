#!/usr/bin/env python3
"""
LUMEN Phase 5 — Tests

Tests for API endpoints, inference, triage, and quality.
"""

import sys
import os
import json
import time
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# ─── Triage Tests ───────────────────────────────────────────────

def test_triage_low():
    from src.triage.rules import get_triage
    result = get_triage(0, 0.95)
    assert result["priority"] == "LOW"
    assert result["stage"] == 0

def test_triage_moderate():
    from src.triage.rules import get_triage
    result = get_triage(2, 0.80)
    assert result["priority"] == "MODERATE"

def test_triage_high():
    from src.triage.rules import get_triage
    result = get_triage(3, 0.90)
    assert result["priority"] == "HIGH"

def test_triage_urgent():
    from src.triage.rules import get_triage
    result = get_triage(4, 0.85)
    assert result["priority"] == "URGENT"

def test_triage_low_confidence_note():
    from src.triage.rules import get_triage
    result = get_triage(2, 0.30)
    assert "low" in result["reason"].lower() or "Manual review" in result["reason"]

def test_triage_invalid_stage():
    from src.triage.rules import get_triage
    result = get_triage(99, 0.90)
    assert result["priority"] == "UNKNOWN"

# ─── Quality Tests ──────────────────────────────────────────────

def test_quality_good_image():
    from src.api.quality import analyze_image_quality
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    result = analyze_image_quality(img)
    assert "status" in result
    assert result["status"] in ("GOOD", "ACCEPTABLE", "POOR", "BAD")
    assert result["width"] == 512
    assert result["height"] == 512
    assert isinstance(result["brightness"], float)
    assert isinstance(result["contrast"], float)
    assert isinstance(result["sharpness"], float)

def test_quality_dark_image():
    from src.api.quality import analyze_image_quality
    img = np.zeros((512, 512, 3), dtype=np.uint8) + 5
    result = analyze_image_quality(img)
    assert result["status"] in ("POOR", "BAD")

def test_quality_check_usable():
    from src.api.quality import check_quality_for_inference
    assert check_quality_for_inference({"status": "GOOD"}) == True
    assert check_quality_for_inference({"status": "ACCEPTABLE"}) == True
    assert check_quality_for_inference({"status": "POOR"}) == False
    assert check_quality_for_inference({"status": "BAD"}) == False

# ─── Inference Service Tests ────────────────────────────────────

@pytest.fixture(scope="module")
def inference_service():
    from src.api.inference import LUMENInferenceService
    ckpt = str(Path(__file__).parent.parent / "outputs/phase3/checkpoints/resnet50_transfer/best_model.pt")
    return LUMENInferenceService(ckpt)

def test_model_loads(inference_service):
    assert inference_service is not None
    assert inference_service.model is not None

def test_screen_returns_success(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    result = inference_service.screen(img, generate_gradcam=False)
    assert result["success"] == True

def test_prediction_valid(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    result = inference_service.screen(img, generate_gradcam=False)
    assert 0 <= result["prediction"]["stage"] <= 4
    assert result["prediction"]["label"] in ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    assert 0 < result["prediction"]["confidence"] <= 1.0

def test_probabilities_sum_to_one(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    result = inference_service.screen(img, generate_gradcam=False)
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4

def test_all_five_stages_in_probs(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    result = inference_service.screen(img, generate_gradcam=False)
    for stage in range(5):
        assert str(stage) in result["probabilities"]

def test_gradcam_generation(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    out_dir = str(Path(__file__).parent.parent / "outputs/phase5/test")
    result = inference_service.screen(img, generate_gradcam=True, output_dir=out_dir, image_id="test_gc")
    assert result["explainability"]["heatmap_available"] == True
    assert result["explainability"]["visualization_path"] is not None

def test_triage_present(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    result = inference_service.screen(img, generate_gradcam=False)
    assert "priority" in result["triage"]
    assert result["triage"]["priority"] in ("LOW", "MODERATE", "HIGH", "URGENT", "UNKNOWN")

def test_timing_logged(inference_service):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    img = cv2.imread(str(img_path))
    result = inference_service.screen(img, generate_gradcam=False)
    assert result["inference_time_ms"] > 0

# ─── API Endpoint Tests ────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api import main as api_main
    from src.api.inference import LUMENInferenceService

    # Load model manually since TestClient doesn't trigger lifespan
    ckpt = str(Path(__file__).parent.parent / "outputs/phase3/checkpoints/resnet50_transfer/best_model.pt")
    api_main.service = LUMENInferenceService(ckpt)

    return TestClient(api_main.app)

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_screen_valid_image(client):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    with open(img_path, "rb") as f:
        resp = client.post("/api/v1/screen", files={"file": ("test.png", f, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] == True
    assert 0 <= data["prediction"]["stage"] <= 4

def test_screen_invalid_file_type(client):
    resp = client.post("/api/v1/screen", files={"file": ("test.txt", b"hello", "text/plain")})
    assert resp.status_code == 400

def test_screen_empty_file(client):
    resp = client.post("/api/v1/screen", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400

def test_quality_endpoint(client):
    img_path = Path(__file__).parent.parent / "datasets/aptos2019/train/000c1434d8d7.png"
    with open(img_path, "rb") as f:
        resp = client.post("/api/v1/quality", files={"file": ("test.png", f, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "brightness" in data
    assert "usable" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
