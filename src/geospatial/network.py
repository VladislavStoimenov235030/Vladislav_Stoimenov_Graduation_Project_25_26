"""Road network loading for the geospatial layer.

Scope decision (agreed with project owner): the routing/geospatial demo
targets a single real area -- Sofia, Bulgaria -- rather than all of
Bulgaria, since the product is framed as a municipality tool. A full
country-level drivable network would be hundreds of thousands of edges and
slow to download/route over; a city-radius graph around Sofia is small,
fast, and still contains one of our two real geotagged test photos
(IMG20260728223743.jpg, ~3.3km from Sofia's center).

The graph is downloaded once via OpenStreetMap (through osmnx) and cached
to disk as GraphML, so repeated pipeline runs (and the Streamlit app) don't
re-hit the OSM Overpass API every time.
"""

from __future__ import annotations

from pathlib import Path

import osmnx as ox

# Sofia city center (Sveta Nedelya square area).
SOFIA_CENTER = (42.6977, 23.3219)
SOFIA_RADIUS_METERS = 8000  # ~8km radius, covers central Sofia + our test point

CACHE_PATH = Path(__file__).resolve().parents[2] / "outputs" / "geo" / "sofia_drive_network.graphml"


def get_sofia_graph(force_download: bool = False) -> "ox.MultiDiGraph":
    """Load the Sofia drivable road network, downloading and caching it on first use."""
    if CACHE_PATH.exists() and not force_download:
        return ox.load_graphml(CACHE_PATH)

    graph = ox.graph_from_point(
        SOFIA_CENTER,
        dist=SOFIA_RADIUS_METERS,
        network_type="drive",
        simplify=True,
    )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, CACHE_PATH)
    return graph


if __name__ == "__main__":
    graph = get_sofia_graph()
    print(f"Sofia drive graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Cached at: {CACHE_PATH}")
