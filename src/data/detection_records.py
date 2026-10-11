"""Structured detection records.

Joins the detection module (src/detection/detector.py) and the GPS module
(src/data/gps_utils.py) into one structured record per detection. This is
the first artifact in the pipeline that gets written to disk -- everything
downstream (risk scoring, geospatial mapping, routing, Streamlit) reads
from this table instead of calling the detector/GPS modules directly.

road_segment_id is intentionally left blank here -- it is filled in by the
geospatial matching step, once road segments exist to match against.
risk_score is filled in using src.risk.scoring (per-detection score only;
segment-level aggregation happens later, in the geospatial/routing step).
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data.gps_utils import extract_gps
from src.detection.detector import DamageDetector, Detection
from src.risk.scoring import detection_risk_score

FIELDNAMES = [
    "detection_id",
    "image_id",
    "latitude",
    "longitude",
    "timestamp",
    "damage_class_id",
    "damage_type",
    "confidence",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "road_segment_id",
    "risk_score",
    "status",
]

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "detections" / "pipeline_test_detections.csv"
)


def detections_to_records(image_path: str | Path, detections: list[Detection]) -> list[dict]:
    """Join one image's detections with its GPS coordinate into structured record dicts."""
    image_path = Path(image_path)
    gps = extract_gps(image_path)

    records = []
    for i, det in enumerate(detections):
        records.append(
            {
                "detection_id": f"{image_path.stem}_{i}",
                "image_id": image_path.name,
                "latitude": gps.latitude if gps else "",
                "longitude": gps.longitude if gps else "",
                "timestamp": gps.timestamp if gps else "",
                "damage_class_id": det.class_id,
                "damage_type": det.class_name,
                "confidence": round(det.confidence, 4),
                "bbox_xmin": det.bbox_xyxy[0],
                "bbox_ymin": det.bbox_xyxy[1],
                "bbox_xmax": det.bbox_xyxy[2],
                "bbox_ymax": det.bbox_xyxy[3],
                "road_segment_id": "",  # filled by geospatial step
                "risk_score": detection_risk_score(det.class_name, det.confidence),
                "status": "",
            }
        )
    return records


def build_records_for_folder(
    image_dir: str | Path, detector: DamageDetector | None = None
) -> list[dict]:
    """Run detection + GPS extraction on every image in a folder, return all records."""
    image_dir = Path(image_dir)
    detector = detector or DamageDetector()

    all_records: list[dict] = []
    for image_path in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))):
        detections = detector.predict(image_path)
        all_records.extend(detections_to_records(image_path, detections))
    return all_records


def write_records_csv(records: list[dict], output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    return output_path


if __name__ == "__main__":
    test_dir = Path(__file__).resolve().parents[2] / "data" / "pipeline_test"
    records = build_records_for_folder(test_dir)
    out_path = write_records_csv(records)
    print(f"Wrote {len(records)} detection record(s) to {out_path}")
    for r in records:
        print(
            f"   {r['image_id']} | {r['damage_type']} | conf={r['confidence']} | "
            f"lat={r['latitude']} lon={r['longitude']}"
        )
