"""Safer Route Planner page.

Wraps src/routing/router.py: origin/destination selection, shortest vs
risk-aware route comparison, both routes on the map, and a plain-language
decision message (spec Section 6-7).

Note on scope (documented, not silently done): this page uses our existing
additive-cost single-alpha router (length + alpha * risk_density), not the
Streamlit-app-spec's candidate-generation + hard 10% cap approach -- that
rework was explicitly deferred earlier in this project's build (see
conversation log / src/routing/router.py docstring). constraint_satisfied
is therefore not computed here.
"""

from __future__ import annotations

import time

import folium
import streamlit as st
from streamlit_folium import st_folium

from src.app import database as db
from src.app.resources import get_graph
from src.app.sidebar import render_sidebar
from src.app.theme import page_header
from src.config import load_app_config, load_risk_weights
from src.geospatial.geocoding import geocode_address
from src.routing.router import compare_routes

st.set_page_config(page_title="Road Guard AI · Safer Route Planner", page_icon="🧭", layout="wide")
render_sidebar()
db.init_db()

page_header(
    "🧭",
    "Safer Route Planner",
    "Compare the distance-based shortest route with a risk-aware alternative within the Sofia pilot area.",
)

pilot_area = load_app_config()["pilot_area"]
default_alpha = load_risk_weights()["routing_alpha"]

PRESET_ROUTES = {
    "Custom": None,
    "Test route (crossroad -> south)": ((42.714282, 23.298205), (42.703906, 23.293455)),
}

st.markdown(f"**Pilot area:** {pilot_area['name']} &nbsp;|&nbsp; **Travel mode:** driving")

preset = st.selectbox("Predefined test route", options=list(PRESET_ROUTES.keys()))
col1, col2 = st.columns(2)

# PINS_KEY holds a fixed 2-slot list [origin_or_None, destination_or_None].
#
# Design note (after several failed attempts at keeping origin/destination in
# live number_input widgets that get updated by clicks/geocoding): a real
# browser can fail to visually refresh an already-rendered widget when its
# session_state is updated purely from server-side code -- invisible in
# headless testing, which doesn't run the real frontend, and confirmed
# directly during this project's build. Rather than keep fighting that
# widget-refresh class of bug, once a pin is placed its coordinates are
# never put back into a stateful widget at all -- they're read fresh from
# session_state and shown as plain text every render, which cannot desync.
# Editable number_input fields are only shown for a slot that has no pin yet.
PINS_KEY = "route_pins"
if PINS_KEY not in st.session_state or preset != st.session_state.get("route_pins_preset"):
    st.session_state[PINS_KEY] = [None, None]
    st.session_state["route_pins_preset"] = preset
pins = st.session_state[PINS_KEY]


def _address_lookup_row(label: str, idx: int) -> None:
    """Address text input + Locate button that places a pin (spec 6.1:
    "Enter an address, if geocoding is implemented"). Geocoding an address
    is treated exactly like clicking that spot on the map -- it fills the
    same pin slot."""
    addr_col, btn_col = st.columns([4, 1])
    address = addr_col.text_input(
        f"{label} address", key=f"addr_{idx}_{preset}", placeholder="e.g. ул. Крум Стоянов"
    )
    if btn_col.button("📍 Locate", key=f"locate_{idx}_{preset}", width="stretch") and address:
        result = geocode_address(address)
        if result is None:
            st.error(
                f"Could not find '{address}'. Try a more specific query, or enter coordinates directly.",
                icon="⚠️",
            )
        else:
            if result.far_from_pilot_area:
                st.warning(
                    f"'{address}' geocoded to {result.latitude:.5f}, {result.longitude:.5f}, "
                    f"which is well outside the Sofia pilot area -- double-check this is the right place.",
                    icon="⚠️",
                )
            st.session_state[PINS_KEY][idx] = (result.latitude, result.longitude)
            st.rerun()


if PRESET_ROUTES[preset] is not None:
    (origin_lat, origin_lon), (dest_lat, dest_lon) = PRESET_ROUTES[preset]
    with col1:
        st.markdown("**Origin**")
        st.info(f"📍 {origin_lat:.6f}, {origin_lon:.6f}")
    with col2:
        st.markdown("**Destination**")
        st.info(f"📍 {dest_lat:.6f}, {dest_lon:.6f}")
else:
    target_idx = 0 if pins[0] is None else (1 if pins[1] is None else None)

    with st.expander("📍 Or select origin/destination on a map", expanded=False):
        if target_idx == 0:
            st.caption("Click anywhere on the map below to place the **origin** pin.")
        elif target_idx == 1:
            st.caption("Origin pin placed. Click again to place the **destination** pin.")
        else:
            st.caption(
                "Both pins placed -- their coordinates are shown below. Use **Clear pins** to start over."
            )

        if st.button("🗑️ Clear pins", disabled=target_idx == 0):
            st.session_state[PINS_KEY] = [None, None]
            st.rerun()

        click_map = folium.Map(
            location=(pilot_area["center_lat"], pilot_area["center_lon"]), zoom_start=13
        )
        pin_labels = [("Origin", "blue", "play"), ("Destination", "black", "flag")]
        for pin, (label, color, icon) in zip(pins, pin_labels):
            if pin is not None:
                folium.Marker(
                    list(pin), tooltip=label, icon=folium.Icon(color=color, icon=icon)
                ).add_to(click_map)

        # Keying on how many pins are placed forces this component to fully
        # remount (fresh last_clicked=None) each time a pin is placed, so it
        # can't resend a stale click on an unrelated rerun (e.g. pressing
        # "Calculate routes") -- st_folium's underlying component otherwise
        # keeps returning its last known click value across reruns.
        n_placed = sum(p is not None for p in pins)
        map_data = st_folium(
            click_map,
            height=400,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key=f"route_click_map_{n_placed}",
        )

        last_clicked = (map_data or {}).get("last_clicked")
        if last_clicked and target_idx is not None:
            clicked = (round(last_clicked["lat"], 6), round(last_clicked["lng"], 6))
            st.session_state[PINS_KEY][target_idx] = clicked
            st.rerun()

    with col1:
        st.markdown("**Origin**")
        if pins[0] is not None:
            origin_lat, origin_lon = pins[0]
            st.success(f"📍 {origin_lat:.6f}, {origin_lon:.6f}")
        else:
            _address_lookup_row("Origin", 0)
            origin_lat = st.number_input(
                "Origin latitude",
                value=pilot_area["center_lat"],
                format="%.6f",
                key=f"manual_origin_lat_{preset}",
            )
            origin_lon = st.number_input(
                "Origin longitude",
                value=pilot_area["center_lon"],
                format="%.6f",
                key=f"manual_origin_lon_{preset}",
            )
    with col2:
        st.markdown("**Destination**")
        if pins[1] is not None:
            dest_lat, dest_lon = pins[1]
            st.success(f"📍 {dest_lat:.6f}, {dest_lon:.6f}")
        else:
            _address_lookup_row("Destination", 1)
            dest_lat = st.number_input(
                "Destination latitude",
                value=pilot_area["center_lat"],
                format="%.6f",
                key=f"manual_dest_lat_{preset}",
            )
            dest_lon = st.number_input(
                "Destination longitude",
                value=pilot_area["center_lon"],
                format="%.6f",
                key=f"manual_dest_lon_{preset}",
            )

alpha = st.slider(
    "Risk-aversion strength (alpha)",
    min_value=0,
    max_value=40000,
    value=default_alpha,
    step=500,
    help="Higher alpha makes the risk-aware route avoid damaged segments more aggressively, "
    "at the cost of extra distance. Default (15000) was found empirically against real pilot data.",
)

calculate = st.button("🧭 Calculate routes", type="primary")

if calculate:
    graph = get_graph()
    try:
        comparison = compare_routes(
            graph, (origin_lat, origin_lon), (dest_lat, dest_lon), alpha=alpha
        )
    except Exception as exc:  # noqa: BLE001 -- user-facing message, not a traceback
        st.error(f"Could not calculate a route between these points: {exc}", icon="⚠️")
        st.stop()

    shortest = comparison.shortest_route
    risk_aware = comparison.risk_aware_route
    distance_increase_pct = (
        round((risk_aware.distance_m - shortest.distance_m) / shortest.distance_m * 100, 1)
        if shortest.distance_m > 0
        else 0.0
    )
    exposure_reduction_pct = (
        round(
            (shortest.total_risk_encountered - risk_aware.total_risk_encountered)
            / shortest.total_risk_encountered
            * 100,
            1,
        )
        if shortest.total_risk_encountered > 0
        else 0.0
    )

    # --- Map (spec 6.7): origin/destination markers, shortest in blue, safer in green ---
    center = ((origin_lat + dest_lat) / 2, (origin_lon + dest_lon) / 2)
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")
    folium.Marker(
        location=[origin_lat, origin_lon],
        tooltip="Origin",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(fmap)
    folium.Marker(
        location=[dest_lat, dest_lon],
        tooltip="Destination",
        icon=folium.Icon(color="black", icon="flag"),
    ).add_to(fmap)
    folium.PolyLine(
        shortest.coordinates, color="blue", weight=5, opacity=0.8, tooltip="Shortest route"
    ).add_to(fmap)
    folium.PolyLine(
        risk_aware.coordinates,
        color="green",
        weight=5,
        opacity=0.6,
        dash_array="8",
        tooltip="Risk-aware (safer) route",
    ).add_to(fmap)
    if shortest.segment_ids == risk_aware.segment_ids:
        st.info(
            "The shortest and risk-aware routes are identical for this origin/destination/alpha -- "
            "there was no known-damaged segment worth detouring around.",
            icon="ℹ️",
        )
    st_folium(fmap, height=500, use_container_width=True, returned_objects=[])

    # --- Comparison table (spec Section 7) ---
    st.subheader("Route comparison")
    st.table(
        {
            "Metric": ["Distance", "Hazard segments encountered", "Total risk encountered"],
            "Shortest route": [
                f"{shortest.distance_m / 1000:.2f} km",
                str(shortest.n_hazard_segments),
                str(shortest.total_risk_encountered),
            ],
            "Safer route": [
                f"{risk_aware.distance_m / 1000:.2f} km",
                str(risk_aware.n_hazard_segments),
                str(risk_aware.total_risk_encountered),
            ],
            "Difference": [
                f"{'+' if distance_increase_pct >= 0 else ''}{distance_increase_pct}%",
                str(risk_aware.n_hazard_segments - shortest.n_hazard_segments),
                f"{'+' if exposure_reduction_pct <= 0 else '-'}{abs(exposure_reduction_pct)}%",
            ],
        }
    )

    # --- Decision message (spec Section 7) ---
    if shortest.segment_ids == risk_aware.segment_ids:
        st.success(
            "No alternative route was needed -- the shortest route already avoids all known road damage."
        )
    elif exposure_reduction_pct > 0:
        st.success(
            f"The safer route reduces known road-damage exposure by {exposure_reduction_pct}% "
            f"while increasing the journey distance by {distance_increase_pct}%."
        )
    else:
        st.warning(
            "The alternative route is longer but does not provide a meaningful reduction in hazard exposure. "
            "The shortest route remains recommended."
        )

    st.caption(
        "Note: this prototype does not yet enforce a hard maximum distance-increase cap on the safer route "
        "(deferred from the full application spec) -- `alpha` is the only control on how much extra distance "
        "the risk-aware route is allowed to add."
    )

    db.insert_route(
        {
            "route_id": f"route_{int(time.time())}",
            "origin_latitude": origin_lat,
            "origin_longitude": origin_lon,
            "destination_latitude": dest_lat,
            "destination_longitude": dest_lon,
            "alpha_used": alpha,
            "shortest_distance": shortest.distance_m,
            "safe_distance": risk_aware.distance_m,
            "distance_increase_percentage": distance_increase_pct,
            "shortest_exposure": shortest.total_risk_encountered,
            "safe_exposure": risk_aware.total_risk_encountered,
            "exposure_reduction_percentage": exposure_reduction_pct,
            "constraint_satisfied": None,
            "calculation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }
    )
