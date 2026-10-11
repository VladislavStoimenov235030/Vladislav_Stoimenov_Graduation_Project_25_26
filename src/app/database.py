"""SQLite persistence layer for the Streamlit application.

Schema follows Streamlit_application_instructions.md Section 9, adapted to
match the field names our pipeline modules already produce (e.g.
`src.risk.observation_risk`, `src.risk.segment_aggregation`,
`src.routing.router`). This module only defines the schema and basic
read/write helpers -- it does not call the detector/risk/routing modules
itself; the Streamlit pages will (write-on-confirm, per spec Section 9).

Four tables:
- observations: one row per confirmed, saved image (spec 9.1 "Images or
  observations table")
- detections: one row per accepted detection within an observation (9.2)
- road_segments: one row per matched OSM edge, aggregated risk (9.3)
- routes: one row per calculated shortest-vs-risk-aware comparison (9.4)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import load_app_config

DB_PATH = Path(__file__).resolve().parents[2] / load_app_config()["paths"]["database"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    image_id              TEXT PRIMARY KEY,
    filename               TEXT NOT NULL,
    file_hash               TEXT NOT NULL UNIQUE,
    image_path               TEXT NOT NULL,
    annotated_image_path      TEXT,
    latitude                   REAL NOT NULL,
    longitude                   REAL NOT NULL,
    coordinate_source            TEXT NOT NULL,  -- 'EXIF' | 'manual' | 'map_selection'
    upload_timestamp               TEXT NOT NULL,
    model_version                    TEXT,
    confidence_threshold               REAL,
    inference_time_ms                    REAL,
    image_risk_score                       REAL,  -- normalized [0,1], src.risk.observation_risk
    risk_category                            TEXT,  -- 'low' | 'medium' | 'high'
    matched_edge_u                             INTEGER,
    matched_edge_v                               INTEGER,
    matched_edge_key                               INTEGER,
    matching_distance_m                             REAL
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id       TEXT PRIMARY KEY,
    image_id             TEXT NOT NULL REFERENCES observations(image_id),
    class_id               INTEGER NOT NULL,
    class_code                TEXT NOT NULL,  -- D00 | D10 | D20 | D40
    class_name                   TEXT NOT NULL,
    confidence                      REAL NOT NULL,
    bbox_x1 REAL NOT NULL, bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL, bbox_y2 REAL NOT NULL,
    relative_bbox_area                  REAL,  -- displayed per spec 2.6; NOT used in risk formula (see src/risk/scoring.py)
    risk_contribution                     REAL NOT NULL  -- src.risk.scoring.detection_risk_score
);

CREATE TABLE IF NOT EXISTS road_segments (
    edge_u                INTEGER NOT NULL,
    edge_v                  INTEGER NOT NULL,
    edge_key                  INTEGER NOT NULL,
    road_name                    TEXT,
    edge_length                     REAL NOT NULL,
    mean_risk                          REAL,
    maximum_risk                          REAL,
    aggregated_risk                          REAL NOT NULL,  -- total_risk, src.risk.segment_aggregation
    risk_density                                REAL NOT NULL,
    risk_category                                  TEXT,
    observation_count                                 INTEGER NOT NULL,
    last_updated                                        TEXT NOT NULL,
    PRIMARY KEY (edge_u, edge_v, edge_key)
);

CREATE TABLE IF NOT EXISTS routes (
    route_id                  TEXT PRIMARY KEY,
    origin_latitude              REAL NOT NULL,
    origin_longitude                REAL NOT NULL,
    destination_latitude               REAL NOT NULL,
    destination_longitude                 REAL NOT NULL,
    alpha_used                               REAL NOT NULL,
    shortest_distance                           REAL NOT NULL,
    safe_distance                                  REAL NOT NULL,
    distance_increase_percentage                      REAL NOT NULL,
    shortest_exposure                                    REAL NOT NULL,
    safe_exposure                                           REAL NOT NULL,
    exposure_reduction_percentage                              REAL,
    constraint_satisfied                                          INTEGER,  -- NULL: 10% cap not enforced yet (deferred, see conversation log)
    calculation_timestamp                                            TEXT NOT NULL
);
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


# --- observations -----------------------------------------------------------


def file_hash_exists(file_hash: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM observations WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row is not None


def insert_observation(observation: dict) -> None:
    columns = (
        "image_id, filename, file_hash, image_path, annotated_image_path, latitude, longitude, "
        "coordinate_source, upload_timestamp, model_version, confidence_threshold, inference_time_ms, "
        "image_risk_score, risk_category, matched_edge_u, matched_edge_v, matched_edge_key, matching_distance_m"
    )
    placeholders = ", ".join("?" for _ in columns.split(","))
    values = tuple(observation.get(col.strip()) for col in columns.split(","))
    with get_connection() as conn:
        conn.execute(f"INSERT INTO observations ({columns}) VALUES ({placeholders})", values)


def get_all_observations() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM observations ORDER BY upload_timestamp DESC").fetchall()


def count_observations() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]


# --- detections --------------------------------------------------------------


def insert_detections(detections: list[dict]) -> None:
    if not detections:
        return
    columns = (
        "detection_id, image_id, class_id, class_code, class_name, confidence, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2, relative_bbox_area, risk_contribution"
    )
    placeholders = ", ".join("?" for _ in columns.split(","))
    rows = [tuple(d.get(col.strip()) for col in columns.split(",")) for d in detections]
    with get_connection() as conn:
        conn.executemany(f"INSERT INTO detections ({columns}) VALUES ({placeholders})", rows)


def get_all_detections() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM detections").fetchall()


def count_detections() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]


# --- road_segments -------------------------------------------------------------


def upsert_road_segment(segment: dict) -> None:
    columns = (
        "edge_u, edge_v, edge_key, road_name, edge_length, mean_risk, maximum_risk, "
        "aggregated_risk, risk_density, risk_category, observation_count, last_updated"
    )
    placeholders = ", ".join("?" for _ in columns.split(","))
    values = tuple(segment.get(col.strip()) for col in columns.split(","))
    with get_connection() as conn:
        conn.execute(
            f"INSERT INTO road_segments ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(edge_u, edge_v, edge_key) DO UPDATE SET "
            f"road_name=excluded.road_name, edge_length=excluded.edge_length, "
            f"mean_risk=excluded.mean_risk, maximum_risk=excluded.maximum_risk, "
            f"aggregated_risk=excluded.aggregated_risk, risk_density=excluded.risk_density, "
            f"risk_category=excluded.risk_category, observation_count=excluded.observation_count, "
            f"last_updated=excluded.last_updated",
            values,
        )


def get_all_road_segments() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM road_segments ORDER BY aggregated_risk DESC").fetchall()


def get_detection_risks_for_edge(u: int, v: int, key: int) -> list[float]:
    """All risk_contribution values from detections whose observation matched this edge."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT d.risk_contribution FROM detections d "
            "JOIN observations o ON d.image_id = o.image_id "
            "WHERE o.matched_edge_u = ? AND o.matched_edge_v = ? AND o.matched_edge_key = ?",
            (u, v, key),
        ).fetchall()
        return [row["risk_contribution"] for row in rows]


def get_latest_timestamp_for_edge(u: int, v: int, key: int) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(upload_timestamp) AS latest FROM observations "
            "WHERE matched_edge_u = ? AND matched_edge_v = ? AND matched_edge_key = ?",
            (u, v, key),
        ).fetchone()
        return row["latest"] if row else None


# --- routes --------------------------------------------------------------------


def insert_route(route: dict) -> None:
    columns = (
        "route_id, origin_latitude, origin_longitude, destination_latitude, destination_longitude, "
        "alpha_used, shortest_distance, safe_distance, distance_increase_percentage, "
        "shortest_exposure, safe_exposure, exposure_reduction_percentage, constraint_satisfied, "
        "calculation_timestamp"
    )
    placeholders = ", ".join("?" for _ in columns.split(","))
    values = tuple(route.get(col.strip()) for col in columns.split(","))
    with get_connection() as conn:
        conn.execute(f"INSERT INTO routes ({columns}) VALUES ({placeholders})", values)


def get_all_routes() -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM routes ORDER BY calculation_timestamp DESC").fetchall()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
    print(f"observations: {count_observations()}, detections: {count_detections()}")
