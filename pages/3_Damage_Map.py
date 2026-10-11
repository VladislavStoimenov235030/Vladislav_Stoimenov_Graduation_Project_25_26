"""Damage Map page: observation markers, road-segment risk layer, heatmap,
and filters (spec Section 4). Filters affect visualization only, per spec
4.2 -- they don't change what's stored or what the router uses.
"""

from __future__ import annotations

from collections import defaultdict

import streamlit as st
from streamlit_folium import st_folium

from src.app import database as db
from src.app.map_utils import build_damage_map
from src.app.resources import get_graph
from src.app.sidebar import render_sidebar
from src.app.theme import page_header
from src.config import load_app_config

st.set_page_config(page_title="Road Guard AI · Damage Map", page_icon="🗺️", layout="wide")
render_sidebar()
db.init_db()

page_header(
    "🗺️",
    "Damage Map",
    "Every confirmed observation and its effect on the surrounding road network.",
)

observations = db.get_all_observations()
if not observations:
    st.info(
        "No saved observations yet. Upload and confirm road images on the "
        "**Upload & Detect** page to populate this map.",
        icon="ℹ️",
    )
    st.stop()

all_detections = db.get_all_detections()
detections_by_image: dict[str, list] = defaultdict(list)
for det in all_detections:
    detections_by_image[det["image_id"]].append(det)

# --- Filters (spec 4.2) ------------------------------------------------------
with st.expander("🔍 Filters", expanded=True):
    col1, col2, col3, col4 = st.columns(4)

    all_classes = sorted({d["class_name"] for d in all_detections})
    with col1:
        selected_classes = st.multiselect(
            "Damage class", options=all_classes, default=all_classes, key="filter_classes"
        )
    with col2:
        selected_categories = st.multiselect(
            "Risk category",
            options=["low", "medium", "high"],
            default=["low", "medium", "high"],
            key="filter_categories",
        )
    with col3:
        min_confidence = st.slider(
            "Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="filter_min_confidence"
        )
    with col4:
        show_no_damage = st.checkbox(
            "Show no-damage observations", value=True, key="filter_show_no_damage"
        )

filtered_observations = []
for obs in observations:
    dets = detections_by_image.get(obs["image_id"], [])
    if not dets:
        if show_no_damage:
            filtered_observations.append(obs)
        continue
    matching_dets = [
        d for d in dets if d["class_name"] in selected_classes and d["confidence"] >= min_confidence
    ]
    if matching_dets and obs["risk_category"] in selected_categories:
        filtered_observations.append(obs)

st.caption(f"Showing {len(filtered_observations)} of {len(observations)} saved observation(s).")

# --- Map ----------------------------------------------------------------------
segments = db.get_all_road_segments()
graph = get_graph()
pilot_area = load_app_config()["pilot_area"]
center = (pilot_area["center_lat"], pilot_area["center_lon"])

fmap = build_damage_map(filtered_observations, detections_by_image, segments, graph, center)
# Sized generously (and full page width) so a marker popup -- annotated
# photo + detail text -- fits fully inside the map area without the page
# needing to scroll to see the rest of it.
st_folium(fmap, height=850, use_container_width=True, returned_objects=[])

# --- Road segment summary table ------------------------------------------------
if segments:
    st.subheader("🛣️ Road segments with recorded damage")
    st.dataframe(
        [
            {
                "Road": s["road_name"] or "(unnamed)",
                "Segment": f"{s['edge_u']}_{s['edge_v']}_{s['edge_key']}",
                "Length (m)": round(s["edge_length"], 1),
                "Observations": s["observation_count"],
                "Mean risk": s["mean_risk"],
                "Max risk": s["maximum_risk"],
                "Aggregated risk": s["aggregated_risk"],
                "Category": s["risk_category"],
            }
            for s in segments
        ],
        width="stretch",
        hide_index=True,
    )
