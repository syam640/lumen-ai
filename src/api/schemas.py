#!/usr/bin/env python3
"""
LUMEN Phase 5 — API Schemas

Pydantic models for request/response validation.
"""

from typing import Dict, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    stage: int
    label: str
    confidence: float


class QualityResponse(BaseModel):
    status: str
    brightness: float
    contrast: float
    sharpness: float
    width: int
    height: int
    message: str = ""


class TriageResponse(BaseModel):
    priority: str
    reason: str


class ExplainabilityResponse(BaseModel):
    method: str
    target_class: int
    heatmap_available: bool
    visualization_path: Optional[str] = None


class ScreenResponse(BaseModel):
    success: bool
    prediction: PredictionResponse
    probabilities: Dict[str, float]
    quality: QualityResponse
    triage: TriageResponse
    explainability: ExplainabilityResponse
    inference_time_ms: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
