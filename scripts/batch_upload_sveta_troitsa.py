"""Batch-processes the Sveta Troitsa neighborhood test photos through the
exact same pipeline the Streamlit Upload & Detect page uses (see
pages/2_Upload_and_Detect.py's save_all_clicked branch) -- same functions,
same confidence threshold, same detector, same graph matching -- so these
observations are indistinguishable in the database from ones a real user
uploaded through the app.

Skips any file already saved (by content hash, same dedup check the app
itself uses), so this is safe to re-run if more photos are added later.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import database as db
from src.app import upload_pipeline as up
from src.config import load_app_config
from src.detection.detector import DamageDetector
from src.geospatial.network import get_sofia_graph

PIPELINE_TEST_DIR = Path(__file__).resolve().parents[1] / "data" / "pipeline_test"

# The 17 Sveta Troitsa test photos (per conversation) -- filenames only,
# matched against data/pipeline_test/. Excludes IMG20260728223743.jpg
# (older, unrelated single test photo) and Showcase_of_routing_test.jpg
# (pre-existing demo file, not part of this batch).
TEST_ROUTE_FILES = [
    "Test_Route1.jpg",
    "Test_Route2.jpg",
    "Test_Route4.jpg",
    "Test_Route5.jpg",
    "Test_Route6#.jpg",
    "Test_Route_5.jpg",
    "Test_Route_7.jpg",
    "Test_Route_8.jpg",
    "Test_Route_9.jpg",
    "Test_Route_10.jpg",
    "Test_Route_11.jpg",
    "Test_Route_12.jpg",
    "Test_Route_13.jpg",
    "Test_Route_14.jpg",
    "Test_Route_15.jpg",
    "Test_Route_16.jpg",
    "Test_Route_17.jpg",
]


def main():
    db.init_db()
    confidence_threshold = load_app_config()["model"]["default_confidence_threshold"]
    detector = DamageDetector(confidence_threshold=confidence_threshold)
    graph = get_sofia_graph()

    print(f"Detector: {detector.model_path}")
    print(f"Confidence threshold: {confidence_threshold}\n")

    n_saved, n_skipped, n_missing = 0, 0, 0

    for filename in TEST_ROUTE_FILES:
        path = PIPELINE_TEST_DIR / filename
        if not path.exists():
            print(f"{filename}: file not found, skipping")
            n_missing += 1
            continue

        file_bytes = path.read_bytes()
        file_hash = up.compute_file_hash(file_bytes)

        if db.file_hash_exists(file_hash):
            print(f"{filename}: already saved, skipping")
            n_skipped += 1
            continue

        processed = up.process_upload(file_bytes, filename)
        if not processed.can_open:
            print(f"{filename}: could not open ({processed.open_error}), skipping")
            continue
        if processed.latitude is None or processed.longitude is None:
            print(f"{filename}: no GPS found, skipping (needs manual coordinates via the app UI)")
            continue

        detection_result = up.run_detection(detector, processed.image_path)
        accepted = up.accepted_detections(detection_result.detections, confidence_threshold)

        match = up.save_observation(
            image_id=file_hash[:16],
            filename=filename,
            file_hash=file_hash,
            image_path=processed.image_path,
            latitude=processed.latitude,
            longitude=processed.longitude,
            coordinate_source=processed.coordinate_source,
            confidence_threshold=confidence_threshold,
            inference_time_ms=detection_result.inference_time_ms,
            detections=accepted,
            graph=graph,
        )

        status = (
            "in scope, matched"
            if match.in_scope and not match.needs_manual_review
            else (
                "in scope, needs manual review" if match.in_scope else "OUT OF SCOPE (not matched)"
            )
        )
        print(
            f"{filename}: {len(accepted)} detection(s), saved -- {status}"
            + (f" ({match.distance_m}m from nearest road)" if match.distance_m is not None else "")
        )
        n_saved += 1

    print(f"\nSaved {n_saved}, skipped (already saved) {n_skipped}, missing {n_missing}.")
    print(f"Total observations in database now: {db.count_observations()}")


if __name__ == "__main__":
    main()
