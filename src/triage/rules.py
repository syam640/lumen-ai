#!/usr/bin/env python3
"""
LUMEN Phase 5 — Triage Rules

Rule-based triage layer for DR screening demonstration.

IMPORTANT: These are demonstration rules for the SIH prototype.
They are NOT clinically validated referral thresholds.
Actual clinical referral must be reviewed by qualified ophthalmologists.
"""

from typing import Dict, Any


# Stage-to-triage mapping
TRIAGE_RULES = {
    0: {
        "priority": "LOW",
        "reason": "No signs of diabetic retinopathy detected. Routine follow-up recommended.",
    },
    1: {
        "priority": "LOW",
        "reason": "Mild DR detected. Schedule routine ophthalmology review.",
    },
    2: {
        "priority": "MODERATE",
        "reason": "Moderate DR detected. Priority ophthalmology referral recommended.",
    },
    3: {
        "priority": "HIGH",
        "reason": "Severe DR detected. Urgent ophthalmology referral recommended.",
    },
    4: {
        "priority": "URGENT",
        "reason": "Proliferative DR detected. Immediate ophthalmology referral required.",
    },
}

DR_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}


def get_triage(stage: int, confidence: float) -> Dict[str, Any]:
    """Determine triage priority for a given DR stage prediction.

    Args:
        stage: Predicted DR stage (0-4).
        confidence: Model confidence (0-1).

    Returns:
        Dict with priority, reason, and stage info.
    """
    if stage not in TRIAGE_RULES:
        return {
            "priority": "UNKNOWN",
            "reason": f"Invalid stage: {stage}",
            "stage": stage,
            "label": "Unknown",
        }

    rule = TRIAGE_RULES[stage]

    # Low-confidence warning
    low_confidence_note = ""
    if confidence < 0.5:
        low_confidence_note = (
            " Note: Model confidence is low. Manual review strongly recommended."
        )

    return {
        "priority": rule["priority"],
        "reason": rule["reason"] + low_confidence_note,
        "stage": stage,
        "label": DR_LABELS[stage],
    }
