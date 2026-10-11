"""Shared "max + damped extra" risk combinator.

Used identically at two different levels of the pipeline:
- image/observation level (src/risk/observation_risk.py): combining
  multiple detections within one photo.
- road-segment level (src/risk/segment_aggregation.py): combining
  detections matched to the same OSM edge.

Both use the same reasoning, so the formula lives in one place: the single
worst hazard should dominate the combined score (a real pothole should not
be diluted by several hairline cracks), but additional co-occurring damage
should still add to the score rather than being ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import load_risk_weights

BETA: float = load_risk_weights()["beta"]


@dataclass
class CombinedRisk:
    max_component: float
    damped_component: float
    raw_total: float
    n_values: int


def combine_risks(risks: list[float], beta: float = BETA) -> CombinedRisk | None:
    """raw_total = max(risks) + beta * sum(remaining risks)."""
    if not risks:
        return None
    sorted_risks = sorted(risks, reverse=True)
    max_component = sorted_risks[0]
    damped_component = round(beta * sum(sorted_risks[1:]), 4)
    raw_total = round(max_component + damped_component, 4)
    return CombinedRisk(
        max_component=max_component,
        damped_component=damped_component,
        raw_total=raw_total,
        n_values=len(risks),
    )
