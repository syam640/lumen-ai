#!/usr/bin/env python3
"""
LUMEN Phase 5 — FastAPI Application

Endpoints:
  GET  /health          — health check
  POST /api/v1/screen   — full screening pipeline
  POST /api/v1/quality  — quality check only
"""

import io
import os
import sys
import time
import logging
import base64
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    ScreenResponse, HealthResponse, PredictionResponse,
    QualityResponse, TriageResponse, ExplainabilityResponse,
)
from src.api.inference import LUMENInferenceService
from src.api.quality import analyze_image_quality, check_quality_for_inference

logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
CHECKPOINT_PATH = os.environ.get(
    "MODEL_PATH",
    str(BASE_DIR / "outputs/phase3/checkpoints/resnet50_transfer/best_model.pt"),
)
GRADCAM_OUTPUT_DIR = str(BASE_DIR / "outputs/phase5/visualizations")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Auto-download model if not found locally
if not Path(CHECKPOINT_PATH).exists():
    try:
        import subprocess
        import urllib.request
        logger.info(f"Model not found at {CHECKPOINT_PATH}, attempting download from Google Drive...")
        Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
        gdrive_file_id = os.environ.get("MODEL_GDRIVE_ID", "1Nv-3JXWhKG5Z0kSsRGQ9IKiqGGtzWKZG")
        gdrive_url = f"https://drive.google.com/uc?export=download&id={gdrive_file_id}"
        # Download with confirmation handling for large files
        session = urllib.request.Session()
        response = session.get(gdrive_url, stream=True)
        # Check for virus scan confirmation page
        if b"confirm=" in response.content or b"download_warning" in response.content:
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    gdrive_url = f"{gdrive_url}&confirm={value}"
                    response = session.get(gdrive_url, stream=True)
                    break
        with open(CHECKPOINT_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Model downloaded to {CHECKPOINT_PATH} ({Path(CHECKPOINT_PATH).stat().st_size / 1e6:.1f}MB)")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")

# CORS configuration
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
CORS_ORIGINS = [origin.strip() for origin in FRONTEND_URL.split(",") if origin.strip()]

# Global service instance
service: LUMENInferenceService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global service
    logger.info("Loading LUMEN model...")
    service = LUMENInferenceService(CHECKPOINT_PATH)
    logger.info("LUMEN model ready.")
    yield
    logger.info("LUMEN shutting down.")


app = FastAPI(
    title="LUMEN API",
    description="AI-assisted Diabetic Retinopathy Screening Research Prototype",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (production build)
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

# Mount visualizations for serving generated images
vis_dir = Path(GRADCAM_OUTPUT_DIR)
vis_dir.mkdir(parents=True, exist_ok=True)
app.mount("/visualizations", StaticFiles(directory=str(vis_dir)), name="visualizations")


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Read image bytes into BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. File may be corrupted or unsupported.")
    return img


def _image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64 data URL."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=service is not None,
        version="0.1.0",
    )


@app.post("/api/v1/screen")
async def screen(file: UploadFile = File(...)):
    """Full screening pipeline: quality check + inference + Grad-CAM + triage."""
    t0 = time.time()

    # Validate file
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB.")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Decode image
    try:
        image_bgr = _read_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate image ID from filename
    image_id = Path(file.filename).stem

    # Run screening
    result = service.screen(
        image_bgr,
        generate_gradcam=True,
        output_dir=GRADCAM_OUTPUT_DIR,
        image_id=image_id,
    )

    # Add visualization URL and base64 data
    if result["explainability"]["visualization_path"]:
        vis_path = result["explainability"]["visualization_path"]
        vis_filename = Path(vis_path).name
        result["explainability"]["visualization_url"] = f"/visualizations/{vis_filename}"
        # Also include base64 for platforms without persistent filesystem
        try:
            result["explainability"]["visualization_base64"] = _image_to_base64(vis_path)
        except Exception:
            pass

    total_time = (time.time() - t0) * 1000
    result["total_time_ms"] = total_time

    return result


@app.post("/api/v1/quality")
async def quality_check(file: UploadFile = File(...)):
    """Image quality check only (no model inference)."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB.")

    try:
        image_bgr = _read_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    quality = analyze_image_quality(image_bgr)
    quality["usable"] = check_quality_for_inference(quality)

    return quality
