"""Non-UI logic for the Upload & Detect page.

Kept separate from pages/2_Upload_and_Detect.py so it can be exercised and
tested without a running Streamlit session -- the page itself should stay
a thin wrapper around these functions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.app import database as db
from src.config import load_app_config
from src.data.gps_utils import GpsCoordinate, extract_gps
from src.detection.annotate import relative_bbox_area
from src.detection.detector import DamageDetector, Detection
from src.geospatial.fill_segments import is_in_scope
from src.geospatial.matching import match_point_to_segment_with_distance, parse_segment_id
from src.risk.observation_risk import ObservationRisk, observation_risk_score
from src.risk.scoring import detection_risk_score
from src.risk.segment_aggregation import recompute_segment_from_db

UPLOADS_DIR = Path(__file__).resolve().parents[2] / load_app_config()["paths"]["uploads"]
MAX_MATCHING_DISTANCE_M = load_app_config()["max_matching_distance_m"]


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def save_uploaded_file(file_bytes: bytes, filename: str, file_hash: str) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / f"{file_hash}_{filename}"
    if not dest.exists():
        dest.write_bytes(file_bytes)
    return dest


def validate_coordinates(latitude: float, longitude: float) -> bool:
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def can_open_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, None
    except Exception as exc:  # noqa: BLE001 -- surfaced as a user-facing message, not a traceback
        return False, str(exc)


@dataclass
class ProcessedUpload:
    file_hash: str
    filename: str
    image_path: Path
    already_saved: bool
    can_open: bool
    open_error: str | None
    gps: GpsCoordinate | None
    latitude: float | None
    longitude: float | None
    coordinate_source: str  # "EXIF" | "manual"


def process_upload(file_bytes: bytes, filename: str) -> ProcessedUpload:
    """Everything that can happen before the user reviews/edits coordinates or runs detection."""
    file_hash = compute_file_hash(file_bytes)
    image_path = save_uploaded_file(file_bytes, filename, file_hash)
    already_saved = db.file_hash_exists(file_hash)
    can_open, open_error = can_open_image(image_path)

    gps = extract_gps(image_path) if can_open else None
    if gps is not None:
        latitude, longitude, source = gps.latitude, gps.longitude, "EXIF"
    else:
        latitude, longitude, source = None, None, "manual"

    return ProcessedUpload(
        file_hash=file_hash,
        filename=filename,
        image_path=image_path,
        already_saved=already_saved,
        can_open=can_open,
        open_error=open_error,
        gps=gps,
        latitude=latitude,
        longitude=longitude,
        coordinate_source=source,
    )


@dataclass
class DetectionResult:
    detections: list[Detection]
    inference_time_ms: float
    image_size: tuple[int, int]


def run_detection(detector: DamageDetector, image_path: Path) -> DetectionResult:
    start = time.perf_counter()
    detections = detector.predict(image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    with Image.open(image_path) as img:
        size = img.size
    return DetectionResult(
        detections=detections, inference_time_ms=round(elapsed_ms, 1), image_size=size
    )


def accepted_detections(
    detections: list[Detection], confidence_threshold: float
) -> list[Detection]:
    return [d for d in detections if d.confidence >= confidence_threshold]


def build_result_rows(detections: list[Detection], image_size: tuple[int, int]) -> list[dict]:
    """One row per accepted detection, matching the spec 2.6 example table."""
    width, height = image_size
    rows = []
    for det in detections:
        rows.append(
            {
                "Damage": det.class_name,
                "Confidence": round(det.confidence, 2),
                "Relative box area": f"{relative_bbox_area(det, width, height) * 100:.1f}%",
                "Risk contribution": detection_risk_score(det.class_name, det.confidence),
            }
        )
    return rows


def compute_observation_risk(detections: list[Detection]) -> ObservationRisk | None:
    risks = [detection_risk_score(d.class_name, d.confidence) for d in detections]
    return observation_risk_score(risks)


@dataclass
class SegmentMatch:
    in_scope: bool
    segment_id: str | None
    distance_m: float | None
    needs_manual_review: bool


def match_to_road(graph, latitude: float, longitude: float) -> SegmentMatch:
    """Match a point to the Sofia pilot-area road network, per spec Section 5."""
    lats = [data["y"] for _, data in graph.nodes(data=True)]
    lons = [data["x"] for _, data in graph.nodes(data=True)]
    bounds = (min(lats), max(lats), min(lons), max(lons))

    if not is_in_scope(latitude, longitude, bounds):
        return SegmentMatch(
            in_scope=False, segment_id=None, distance_m=None, needs_manual_review=False
        )

    segment_id, distance_m = match_point_to_segment_with_distance(graph, latitude, longitude)
    needs_review = distance_m > MAX_MATCHING_DISTANCE_M
    return SegmentMatch(
        in_scope=True,
        segment_id=segment_id,
        distance_m=round(distance_m, 1),
        needs_manual_review=needs_review,
    )


def save_observation(
    *,
    image_id: str,
    filename: str,
    file_hash: str,
    image_path: Path,
    latitude: float,
    longitude: float,
    coordinate_source: str,
    confidence_threshold: float,
    inference_time_ms: float,
    detections: list[Detection],
    graph,
) -> SegmentMatch:
    """Persist one confirmed observation + its detections, then recompute the affected segment.

    Matches spec 2.7 ("only saved observations affect downstream views")
    and Section 5 step 6 ("recalculate the aggregated risk of the affected
    edge").
    """
    obs_risk = compute_observation_risk(detections)
    match = match_to_road(graph, latitude, longitude)

    edge_u = edge_v = edge_key = None
    if match.segment_id and not match.needs_manual_review:
        edge_u, edge_v, edge_key = parse_segment_id(match.segment_id)

    db.insert_observation(
        {
            "image_id": image_id,
            "filename": filename,
            "file_hash": file_hash,
            "image_path": str(image_path),
            "annotated_image_path": None,
            "latitude": latitude,
            "longitude": longitude,
            "coordinate_source": coordinate_source,
            "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "model_version": load_app_config()["model"]["version_label"],
            "confidence_threshold": confidence_threshold,
            "inference_time_ms": inference_time_ms,
            "image_risk_score": obs_risk.normalized_score if obs_risk else 0.0,
            "risk_category": obs_risk.category if obs_risk else "low",
            "matched_edge_u": edge_u,
            "matched_edge_v": edge_v,
            "matched_edge_key": edge_key,
            "matching_distance_m": match.distance_m,
        }
    )

    db.insert_detections(
        [
            {
                "detection_id": f"{image_id}_{i}",
                "image_id": image_id,
                "class_id": d.class_id,
                "class_code": d.class_code,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_x1": d.bbox_xyxy[0],
                "bbox_y1": d.bbox_xyxy[1],
                "bbox_x2": d.bbox_xyxy[2],
                "bbox_y2": d.bbox_xyxy[3],
                "relative_bbox_area": None,
                "risk_contribution": detection_risk_score(d.class_name, d.confidence),
            }
            for i, d in enumerate(detections)
        ]
    )

    if edge_u is not None:
        segment = recompute_segment_from_db(edge_u, edge_v, edge_key, graph)
        if segment:
            db.upsert_road_segment(segment)

    return match
