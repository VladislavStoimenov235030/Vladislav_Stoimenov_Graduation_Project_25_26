"""Image-level (observation) risk: combine one image's detections into a
single normalized [0, 1] score plus a low/medium/high category.

This is the piece the Streamlit app spec calls "observation_risk" (section
3.2/3.3) -- distinct from both the per-detection score (src/risk/scoring.py)
and the road-segment aggregate (src/risk/segment_aggregation.py), though it
reuses the same max+damped combinator (src/risk/combine.py) as the segment
level for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import load_risk_weights
from src.risk.combine import combine_risks

_weights = load_risk_weights()
OBSERVATION_RISK_MAX: float = _weights["observation_risk_max"]
LOW_MAX: float = _weights["risk_categories"]["low_max"]
MEDIUM_MAX: float = _weights["risk_categories"]["medium_max"]


@dataclass
class ObservationRisk:
    raw_score: float
    normalized_score: float  # in [0, 1]
    category: str  # "low" | "medium" | "high"


def categorize_risk(normalized_score: float) -> str:
    if normalized_score < LOW_MAX:
        return "low"
    if normalized_score < MEDIUM_MAX:
        return "medium"
    return "high"


def observation_risk_score(detection_risks: list[float]) -> ObservationRisk | None:
    """Combine one image's per-detection risk scores into one normalized observation score.

    Returns None if the image has no detections above the confidence
    threshold -- callers should treat that as a valid "no damage found"
    result (spec 2.6), not as an error or a risk of zero.
    """
    combined = combine_risks(detection_risks)
    if combined is None:
        return None
    normalized = min(combined.raw_total / OBSERVATION_RISK_MAX, 1.0)
    return ObservationRisk(
        raw_score=combined.raw_total,
        normalized_score=round(normalized, 4),
        category=categorize_risk(normalized),
    )


if __name__ == "__main__":
    from src.risk.scoring import detection_risk_score

    scenarios = {
        "one full-confidence pothole": [detection_risk_score("Pothole", 1.0)],
        "one low-confidence crack": [detection_risk_score("Longitudinal Crack", 0.3)],
        "pothole + two cracks": [
            detection_risk_score("Pothole", 0.9),
            detection_risk_score("Longitudinal Crack", 0.6),
            detection_risk_score("Longitudinal Crack", 0.5),
        ],
    }
    for label, risks in scenarios.items():
        result = observation_risk_score(risks)
        print(f"{label:30s} -> {result}")
