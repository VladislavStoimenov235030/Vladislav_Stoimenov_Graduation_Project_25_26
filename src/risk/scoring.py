"""Risk scoring module -- per-detection risk score.

Design (agreed with project owner before implementation, see thesis notes /
conversation log for the full reasoning):

- Severity is fixed per damage class and reflects the hazard *mechanism*,
  not the detected bounding-box size. Bbox area in a 2D photo is a function
  of camera distance/angle, not real-world damage size, so it is
  deliberately NOT used as a risk multiplier. (Re-confirmed as a deliberate
  deviation from the later Streamlit application spec, which suggested a
  size_factor -- kept excluded on purpose.)
- Confidence is used as a mild multiplier, not a linear 0-1 scale. A
  detection below the inference confidence threshold never reaches this
  module at all (it's filtered out at detection time), so every score here
  already represents a "trusted enough" detection -- confidence should
  nudge the score, not swing it 3x between a 0.3 and 0.9 detection of the
  same class.

All numeric weights come from config/risk_weights.yaml (src/config.py) --
this module holds only the formulas, not the numbers.

This module only computes the PER-DETECTION risk score. Image-level and
segment-level aggregation (max + damped extra) live in
src/risk/observation_risk.py and src/risk/segment_aggregation.py, both
built on the shared combinator in src/risk/combine.py.
"""

from __future__ import annotations

from src.config import load_risk_weights

_weights = load_risk_weights()
BASE_SEVERITY: dict[str, float] = _weights["base_severity"]
CONFIDENCE_FLOOR: float = _weights["confidence_floor"]
CONFIDENCE_THRESHOLD: float = _weights["confidence_threshold"]


def confidence_multiplier(confidence: float) -> float:
    """Map a detection confidence to a mild [CONFIDENCE_FLOOR, 1.0] multiplier.

    Confidence gates *whether* a detection is trusted (done upstream at
    inference time); here it only nudges the risk score, since a lower-
    confidence real pothole is not physically less dangerous than a
    high-confidence one -- we're just less sure it's there at all.
    """
    if confidence <= CONFIDENCE_THRESHOLD:
        return CONFIDENCE_FLOOR
    if confidence >= 1.0:
        return 1.0
    span = 1.0 - CONFIDENCE_THRESHOLD
    progress = (confidence - CONFIDENCE_THRESHOLD) / span
    return CONFIDENCE_FLOOR + progress * (1.0 - CONFIDENCE_FLOOR)


def detection_risk_score(damage_type: str, confidence: float) -> float:
    """Per-detection risk score = fixed class severity x mild confidence multiplier."""
    if damage_type not in BASE_SEVERITY:
        raise ValueError(
            f"Unknown damage type '{damage_type}', expected one of {list(BASE_SEVERITY)}"
        )
    return round(BASE_SEVERITY[damage_type] * confidence_multiplier(confidence), 4)


if __name__ == "__main__":
    # Quick manual check across classes and confidence levels.
    for damage_type in BASE_SEVERITY:
        for conf in (0.25, 0.5, 0.9, 1.0):
            print(
                f"{damage_type:25s} conf={conf:.2f} -> risk={detection_risk_score(damage_type, conf)}"
            )
