"""Routing engine: shortest-by-distance vs risk-aware route comparison.

Edge cost for the risk-aware route:

    risk_cost = length_m + alpha * risk_density

risk_density comes from src/risk/segment_aggregation.py (segment_risk.csv)
-- it's already normalized per meter, so a long, evenly-risky road isn't
penalized more than a short one just for being long (see that module's
docstring for why). Segments with no matched detections get risk_density=0,
i.e. no penalty -- absence of data is not treated as evidence of safety in
the thesis narrative, but it IS what "no penalty" means mechanically here,
and worth stating plainly for the same reason.

alpha default: 15000. This was picked empirically against the real Sofia
test data (see conversation/thesis notes): below ~15000 the risk-aware
route still cuts through the single highest-risk known segment because the
penalty doesn't outweigh the extra distance of any detour; at ~15000 it
becomes worthwhile to reroute. This is a rule-based prototype value, not a
calibrated safety threshold -- it should be exposed as an adjustable
parameter (e.g. a Streamlit slider) rather than treated as fixed truth.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import osmnx as ox

from src.config import load_risk_weights
from src.geospatial.network import get_sofia_graph
from src.risk.segment_aggregation import SEGMENT_OUTPUT_PATH

DEFAULT_ALPHA = load_risk_weights()["routing_alpha"]

ROUTE_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "geo" / "route_comparison.json"
)


@dataclass
class RouteResult:
    distance_m: float
    n_hazard_segments: int
    total_risk_encountered: float
    segment_ids: list[str]
    coordinates: list[tuple[float, float]]  # (lat, lon) per node, for map drawing


@dataclass
class RouteComparison:
    origin: tuple[float, float]
    destination: tuple[float, float]
    alpha: float
    shortest_route: RouteResult
    risk_aware_route: RouteResult
    distance_difference_m: float
    risk_reduction: float  # total_risk_encountered(shortest) - total_risk_encountered(risk_aware)


def load_segment_risk(csv_path: str | Path = SEGMENT_OUTPUT_PATH) -> dict[str, dict]:
    """road_segment_id -> {risk_density, total_risk} for every segment with detections."""
    csv_path = Path(csv_path)
    table = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["road_segment_id"]] = {
                "risk_density": float(row["risk_density"]),
                "total_risk": float(row["total_risk"]),
            }
    return table


def _segment_id(u: int, v: int, key: int) -> str:
    return f"{u}_{v}_{key}"


def annotate_risk_cost(
    graph: "ox.MultiDiGraph", segment_risk: dict[str, dict], alpha: float
) -> None:
    """Set a 'risk_cost' attribute on every edge: length_m + alpha * risk_density (0 if no data)."""
    for u, v, key, data in graph.edges(keys=True, data=True):
        seg_id = _segment_id(u, v, key)
        risk_density = segment_risk.get(seg_id, {}).get("risk_density", 0.0)
        data["risk_cost"] = data["length"] + alpha * risk_density


def _path_to_result(
    graph: "ox.MultiDiGraph", path: list[int], segment_risk: dict[str, dict]
) -> RouteResult:
    distance_m = 0.0
    total_risk_encountered = 0.0
    n_hazard_segments = 0
    segment_ids = []

    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        key = next(iter(edge_data))
        data = edge_data[key]
        seg_id = _segment_id(u, v, key)
        segment_ids.append(seg_id)
        distance_m += data["length"]
        if seg_id in segment_risk:
            n_hazard_segments += 1
            total_risk_encountered += segment_risk[seg_id]["total_risk"]

    coordinates = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in path]

    return RouteResult(
        distance_m=round(distance_m, 1),
        n_hazard_segments=n_hazard_segments,
        total_risk_encountered=round(total_risk_encountered, 4),
        segment_ids=segment_ids,
        coordinates=coordinates,
    )


def compare_routes(
    graph: "ox.MultiDiGraph",
    origin_latlon: tuple[float, float],
    dest_latlon: tuple[float, float],
    segment_risk: dict[str, dict] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> RouteComparison:
    """Compute both the shortest-by-distance and risk-aware routes between two points."""
    segment_risk = segment_risk if segment_risk is not None else load_segment_risk()
    annotate_risk_cost(graph, segment_risk, alpha)

    origin_node = ox.distance.nearest_nodes(graph, X=origin_latlon[1], Y=origin_latlon[0])
    dest_node = ox.distance.nearest_nodes(graph, X=dest_latlon[1], Y=dest_latlon[0])

    shortest_path = nx.shortest_path(graph, origin_node, dest_node, weight="length")
    risk_aware_path = nx.shortest_path(graph, origin_node, dest_node, weight="risk_cost")

    shortest_result = _path_to_result(graph, shortest_path, segment_risk)
    risk_aware_result = _path_to_result(graph, risk_aware_path, segment_risk)

    return RouteComparison(
        origin=origin_latlon,
        destination=dest_latlon,
        alpha=alpha,
        shortest_route=shortest_result,
        risk_aware_route=risk_aware_result,
        distance_difference_m=round(risk_aware_result.distance_m - shortest_result.distance_m, 1),
        risk_reduction=round(
            shortest_result.total_risk_encountered - risk_aware_result.total_risk_encountered, 4
        ),
    )


def write_route_comparison(
    comparison: RouteComparison, output_path: str | Path = ROUTE_OUTPUT_PATH
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(comparison), f, indent=2)
    return output_path


if __name__ == "__main__":
    ORIGIN = (42.714282, 23.298205)
    DESTINATION = (42.703906, 23.293455)

    graph = get_sofia_graph()
    comparison = compare_routes(graph, ORIGIN, DESTINATION)
    out_path = write_route_comparison(comparison)

    print(f"alpha = {comparison.alpha}")
    print(
        f"Shortest route:   {comparison.shortest_route.distance_m}m, "
        f"{comparison.shortest_route.n_hazard_segments} hazard segment(s), "
        f"total_risk={comparison.shortest_route.total_risk_encountered}"
    )
    print(
        f"Risk-aware route: {comparison.risk_aware_route.distance_m}m, "
        f"{comparison.risk_aware_route.n_hazard_segments} hazard segment(s), "
        f"total_risk={comparison.risk_aware_route.total_risk_encountered}"
    )
    print(f"Distance difference: {comparison.distance_difference_m}m")
    print(f"Risk reduction: {comparison.risk_reduction}")
    print(f"Written to {out_path}")
