"""Address geocoding for the Safer Route Planner (spec 6.1: "Enter an
address, if geocoding is implemented").

Wraps osmnx's Nominatim-backed geocoder. This is a live network call to
the OSM Nominatim API (same service osmnx uses for the road graph), so it
requires internet access and is subject to Nominatim's rate limits/usage
policy -- fine for occasional interactive lookups in a prototype, not
meant for bulk/batch geocoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import osmnx as ox

from src.config import load_app_config

_pilot_area = load_app_config()["pilot_area"]
PILOT_CENTER = (_pilot_area["center_lat"], _pilot_area["center_lon"])
# Generous sanity-check radius around the pilot area center -- a geocoded
# result further than this is very likely the wrong place (Nominatim
# matched a same-named street/place elsewhere) rather than a real Sofia
# address, so it's flagged rather than silently used.
SANITY_RADIUS_M = 30_000


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [*a, *b])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


@dataclass
class GeocodeResult:
    latitude: float
    longitude: float
    query_used: str
    far_from_pilot_area: bool


def geocode_address(address: str) -> GeocodeResult | None:
    """Geocode a free-text address/place name to (lat, lon).

    Automatically appends the pilot-area name if the query doesn't already
    look like it includes a city, which substantially improves match rate
    for bare street names (e.g. "Krum Hr. Stoyanov" -> "Krum Hr. Stoyanov,
    Sofia, Bulgaria"). Returns None if Nominatim can't find a match at all;
    raises nothing the caller needs to distinguish from "not found" --
    both are reported to the user as "address not found".
    """
    address = address.strip()
    if not address:
        return None

    query = address
    if "sofia" not in address.lower() and "софия" not in address.lower():
        query = f"{address}, {_pilot_area['name']}"

    try:
        lat, lon = ox.geocode(query)
    except Exception:
        return None

    distance = _haversine_m(PILOT_CENTER, (lat, lon))
    return GeocodeResult(
        latitude=lat,
        longitude=lon,
        query_used=query,
        far_from_pilot_area=distance > SANITY_RADIUS_M,
    )


if __name__ == "__main__":
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    for addr in ["ул. Крум Стоянов", "Йосиф Щросмайер", "nonexistent street xyz123"]:
        result = geocode_address(addr)
        print(addr, "->", result)
