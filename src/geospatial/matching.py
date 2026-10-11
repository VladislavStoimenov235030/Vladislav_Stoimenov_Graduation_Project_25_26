"""Nearest-road-segment matching.

Snaps a detection's (lat, lon) to the closest edge in the Sofia road graph.
An OSM graph edge is identified by (u, v, key) -- the two endpoint node IDs
plus a key that disambiguates parallel edges between the same two nodes.
We encode that triple as a single string "u_v_key" to use as
road_segment_id in the detection records table.

Scope: Sofia only (see src/geospatial/network.py). Points outside the
downloaded graph radius (e.g. the Pleven/Ruse-area test photo) are out of
scope for this project's geospatial/routing demo and are not matched here.
"""

from __future__ import annotations

import math

import osmnx as ox


def make_segment_id(u: int, v: int, key: int) -> str:
    return f"{u}_{v}_{key}"


def parse_segment_id(segment_id: str) -> tuple[int, int, int]:
    u, v, key = segment_id.split("_")
    return int(u), int(v), int(key)


def match_point_to_segment(graph: "ox.MultiDiGraph", latitude: float, longitude: float) -> str:
    """Return the road_segment_id of the graph edge nearest to (latitude, longitude)."""
    u, v, key = ox.distance.nearest_edges(graph, X=longitude, Y=latitude)
    return make_segment_id(u, v, key)


def segment_length_meters(graph: "ox.MultiDiGraph", segment_id: str) -> float:
    """Real-world length (meters) of a matched road segment, from OSM edge data."""
    u, v, key = parse_segment_id(segment_id)
    edge_data = graph.get_edge_data(u, v, key)
    return float(edge_data["length"])


def segment_road_name(graph: "ox.MultiDiGraph", segment_id: str) -> str | None:
    u, v, key = parse_segment_id(segment_id)
    edge_data = graph.get_edge_data(u, v, key)
    name = edge_data.get("name")
    if isinstance(name, list):
        name = name[0] if name else None
    return name


def match_point_to_segment_with_distance(
    graph: "ox.MultiDiGraph", latitude: float, longitude: float
) -> tuple[str, float]:
    """Return (road_segment_id, approx_distance_meters) for the nearest edge to (latitude, longitude).

    The graph is unprojected (plain lat/lon), so osmnx's raw nearest_edges
    distance is in degrees, not meters. This converts it to an
    approximate meter distance using a local flat-earth scale factor
    (111,320 m/degree latitude, scaled by cos(latitude) for longitude) --
    accurate enough at Sofia's scale (a few km) to flag "too far from any
    road" cases, not intended as a precise geodesic distance.
    """
    (u, v, key), distance_deg = ox.distance.nearest_edges(
        graph, X=longitude, Y=latitude, return_dist=True
    )
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(latitude))
    scale = (
        meters_per_degree_lat + meters_per_degree_lon
    ) / 2  # rough average, fine for small distances
    distance_m = float(distance_deg) * scale
    return make_segment_id(u, v, key), distance_m


if __name__ == "__main__":
    from src.geospatial.network import get_sofia_graph

    graph = get_sofia_graph()

    # The Sofia-area real test detection.
    lat, lon = 42.712447222222224, 23.295408333333334
    segment_id = match_point_to_segment(graph, lat, lon)
    length = segment_length_meters(graph, segment_id)
    print(f"({lat}, {lon}) -> segment {segment_id}, length {length:.1f}m")
