"""Fill road_segment_id in the detection records CSV.

Only points that actually fall within the Sofia graph's bounding box (plus
a small margin) get matched. This project's geospatial/routing scope is
Sofia only (see src/geospatial/network.py) -- the Pleven/Ruse-area test
photo (~140km from Sofia) is intentionally out of scope and is left with a
blank road_segment_id rather than being force-matched to the nearest (and
meaningless) Sofia edge. osmnx's nearest_edges has no distance cutoff of
its own, so this bounds check is what actually keeps out-of-scope points
out.
"""

from __future__ import annotations

import csv
from pathlib import Path

import osmnx as ox

from src.data.detection_records import DEFAULT_OUTPUT_PATH, FIELDNAMES
from src.geospatial.matching import match_point_to_segment
from src.geospatial.network import get_sofia_graph

# Margin added around the graph's actual node bounding box, in degrees
# (~a few hundred meters), so points right at the edge of the download
# radius aren't rejected purely due to graph-simplification trimming.
BOUNDS_MARGIN_DEG = 0.01


def _graph_bounds(graph: "ox.MultiDiGraph") -> tuple[float, float, float, float]:
    lats = [data["y"] for _, data in graph.nodes(data=True)]
    lons = [data["x"] for _, data in graph.nodes(data=True)]
    return min(lats), max(lats), min(lons), max(lons)


def is_in_scope(lat: float, lon: float, bounds: tuple[float, float, float, float]) -> bool:
    min_lat, max_lat, min_lon, max_lon = bounds
    return (min_lat - BOUNDS_MARGIN_DEG) <= lat <= (max_lat + BOUNDS_MARGIN_DEG) and (
        min_lon - BOUNDS_MARGIN_DEG
    ) <= lon <= (max_lon + BOUNDS_MARGIN_DEG)


def fill_road_segment_ids(csv_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    csv_path = Path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    graph = get_sofia_graph()
    bounds = _graph_bounds(graph)

    for row in rows:
        if row["latitude"] == "" or row["longitude"] == "":
            continue  # no GPS at all, nothing to match
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if not is_in_scope(lat, lon, bounds):
            row["road_segment_id"] = ""  # out of Sofia scope, e.g. Pleven/Ruse test photo
            continue
        row["road_segment_id"] = match_point_to_segment(graph, lat, lon)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


if __name__ == "__main__":
    path = fill_road_segment_ids()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(
                f"{row['image_id']:28s} {row['damage_type']:25s} road_segment_id={row['road_segment_id'] or '(out of scope)'}"
            )
