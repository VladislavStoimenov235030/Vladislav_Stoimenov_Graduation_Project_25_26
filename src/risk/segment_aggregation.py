"""Segment-level risk aggregation.

Combines all per-detection risk scores (src/risk/scoring.py) that share a
road_segment_id (assigned in src/geospatial/fill_segments.py) into two
segment-level numbers, per the design agreed with the project owner. Uses
the shared max+damped combinator (src/risk/combine.py) -- same formula as
the image-level observation score, applied at the segment level instead:

- segment_risk_raw = max(detection risks) + BETA * sum(remaining risks)
  The worst single hazard on the segment dominates the score (a single
  pothole should not be diluted by five hairline cracks), but additional
  co-occurring damage still adds to the score rather than being ignored.

- risk_density = segment_risk_raw / segment_length_meters
  Used for the routing penalty, so a risk-aware route doesn't unfairly
  avoid a road just because it's long and therefore has more total
  detections.

- total_risk = segment_risk_raw (unnormalized)
  Used for the maintenance-priority list, since repair effort scales with
  total damage present on a segment, not damage per meter.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.detection_records import DEFAULT_OUTPUT_PATH
from src.geospatial.matching import segment_length_meters
from src.geospatial.network import get_sofia_graph
from src.risk.combine import combine_risks

SEGMENT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "geo" / "segment_risk.csv"

SEGMENT_FIELDNAMES = [
    "road_segment_id",
    "length_m",
    "n_detections",
    "max_component",
    "damped_component",
    "risk_density",
    "total_risk",
]


@dataclass
class SegmentRisk:
    road_segment_id: str
    length_m: float
    n_detections: int
    max_component: float
    damped_component: float
    risk_density: float
    total_risk: float


def aggregate_segment_risk(detection_risks: list[float], length_m: float) -> SegmentRisk | None:
    """Combine a list of per-detection risk scores on one segment into segment-level numbers."""
    combined = combine_risks(detection_risks)
    if combined is None:
        return None

    risk_density = round(combined.raw_total / length_m, 6) if length_m > 0 else combined.raw_total

    return SegmentRisk(
        road_segment_id="",  # filled by caller
        length_m=length_m,
        n_detections=combined.n_values,
        max_component=combined.max_component,
        damped_component=combined.damped_component,
        risk_density=risk_density,
        total_risk=combined.raw_total,
    )


def build_segment_risk_table(csv_path: str | Path = DEFAULT_OUTPUT_PATH) -> list[SegmentRisk]:
    """Read the detection records CSV, group by road_segment_id, aggregate each group."""
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    risks_by_segment: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        seg_id = row["road_segment_id"]
        if not seg_id or row["risk_score"] == "":
            continue
        risks_by_segment[seg_id].append(float(row["risk_score"]))

    graph = get_sofia_graph()
    segments: list[SegmentRisk] = []
    for seg_id, risks in risks_by_segment.items():
        length_m = segment_length_meters(graph, seg_id)
        result = aggregate_segment_risk(risks, length_m)
        result.road_segment_id = seg_id
        segments.append(result)

    return segments


def write_segment_risk_csv(
    segments: list[SegmentRisk], output_path: str | Path = SEGMENT_OUTPUT_PATH
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_FIELDNAMES)
        writer.writeheader()
        for s in segments:
            writer.writerow(
                {
                    "road_segment_id": s.road_segment_id,
                    "length_m": s.length_m,
                    "n_detections": s.n_detections,
                    "max_component": s.max_component,
                    "damped_component": s.damped_component,
                    "risk_density": s.risk_density,
                    "total_risk": s.total_risk,
                }
            )
    return output_path


def recompute_segment_from_db(u: int, v: int, key: int, graph) -> dict | None:
    """Recompute one segment's aggregate risk from the SQLite detections table (app-facing path).

    Distinct from build_segment_risk_table() above, which reads the
    standalone pipeline's CSV -- this reads live app data (src.app.database)
    and returns a dict ready for db.upsert_road_segment(), including the
    mean_risk/maximum_risk/last_updated fields the app spec (Section 9.3)
    asks for that the CSV-based pipeline path doesn't track.
    """
    import statistics

    from src.app import database as db
    from src.geospatial.matching import make_segment_id, segment_road_name
    from src.risk.observation_risk import OBSERVATION_RISK_MAX, categorize_risk

    risks = db.get_detection_risks_for_edge(u, v, key)
    if not risks:
        return None

    combined = combine_risks(risks)
    length_m = segment_length_meters(graph, make_segment_id(u, v, key))
    risk_density = combined.raw_total / length_m if length_m > 0 else combined.raw_total
    normalized = min(combined.raw_total / OBSERVATION_RISK_MAX, 1.0)

    return {
        "edge_u": u,
        "edge_v": v,
        "edge_key": key,
        "road_name": segment_road_name(graph, make_segment_id(u, v, key)),
        "edge_length": length_m,
        "mean_risk": round(statistics.mean(risks), 4),
        "maximum_risk": max(risks),
        "aggregated_risk": combined.raw_total,
        "risk_density": round(risk_density, 6),
        "risk_category": categorize_risk(normalized),
        "observation_count": len(risks),
        "last_updated": db.get_latest_timestamp_for_edge(u, v, key),
    }


if __name__ == "__main__":
    # Sanity check the formula's behavior on made-up scenarios before trusting it on real data:
    # one pothole vs. three cracks -- the pothole-alone segment should score higher.
    from src.risk.scoring import detection_risk_score

    one_pothole = [detection_risk_score("Pothole", 0.9)]
    three_cracks = [
        detection_risk_score("Longitudinal Crack", 0.9),
        detection_risk_score("Longitudinal Crack", 0.9),
        detection_risk_score("Longitudinal Crack", 0.9),
    ]
    print("Sanity check (length=100m, ignored for total_risk comparison):")
    print("  one pothole      ->", aggregate_segment_risk(one_pothole, length_m=100))
    print("  three crackss    ->", aggregate_segment_risk(three_cracks, length_m=100))

    mixed = [detection_risk_score("Pothole", 0.9), detection_risk_score("Longitudinal Crack", 0.9)]
    print("  pothole + crack  ->", aggregate_segment_risk(mixed, length_m=100))

    print()
    print("Real data (Sofia test segment):")
    segments = build_segment_risk_table()
    out_path = write_segment_risk_csv(segments)
    print(f"Wrote {len(segments)} segment(s) to {out_path}")
    for s in segments:
        print(
            f"   {s.road_segment_id}: n={s.n_detections} length={s.length_m}m "
            f"total_risk={s.total_risk} risk_density={s.risk_density}"
        )
