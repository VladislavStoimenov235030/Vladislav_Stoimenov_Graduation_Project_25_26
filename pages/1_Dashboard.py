"""Dashboard page: summary cards, class/risk breakdowns, recent uploads,
and a small overview map (spec Section "1. Dashboard").
"""

from __future__ import annotations

from collections import Counter, defaultdict

import folium
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from src.app import database as db
from src.app.sidebar import render_sidebar
from src.app.theme import RISK_COLORS, page_header
from src.config import load_app_config

st.set_page_config(page_title="Road Guard AI · Dashboard", page_icon="📊", layout="wide")
render_sidebar()
db.init_db()

page_header("📊", "Dashboard", "Overview of everything detected, scored, and mapped so far.")

observations = db.get_all_observations()

if not observations:
    st.info(
        "No observations yet. Upload geolocated road images on the **Upload & Detect** "
        "page before the road-risk map and route planner have any data to use.",
        icon="ℹ️",
    )
    st.stop()

all_detections = db.get_all_detections()
detections_by_image: dict[str, list] = defaultdict(list)
for det in all_detections:
    detections_by_image[det["image_id"]].append(det)

segments = db.get_all_road_segments()

# --- Summary cards --------------------------------------------------------------
n_with_damage = sum(1 for o in observations if detections_by_image.get(o["image_id"]))
avg_risk = round(sum(o["image_risk_score"] for o in observations) / len(observations), 3)
n_high_risk = sum(1 for o in observations if o["risk_category"] == "high")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Processed images", len(observations))
c2.metric("Detected damages", len(all_detections))
c3.metric("Affected road segments", len(segments))
c4.metric("Average risk score", avg_risk)
c5.metric("High-risk locations", n_high_risk)

st.divider()

col_left, col_right = st.columns(2)

# --- Detections per damage class --------------------------------------------------
with col_left:
    st.subheader("Detections per damage class")
    class_counts = Counter(d["class_name"] for d in all_detections)
    if class_counts:
        fig = px.bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            labels={"x": "Damage class", "y": "Count"},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No detections recorded yet.")

# --- Observations by risk category --------------------------------------------------
with col_right:
    st.subheader("Observations by risk category")
    category_counts = Counter(o["risk_category"] for o in observations)
    order = ["low", "medium", "high"]
    present = [c for c in order if c in category_counts]
    fig = px.pie(
        names=present,
        values=[category_counts[c] for c in present],
        color=present,
        color_discrete_map=RISK_COLORS,
    )
    st.plotly_chart(fig, width="stretch")

st.divider()

# --- Recently processed images --------------------------------------------------------
st.subheader("Recently processed images")
recent = sorted(observations, key=lambda o: o["upload_timestamp"], reverse=True)[:10]
st.dataframe(
    [
        {
            "Filename": o["filename"],
            "Processed": o["upload_timestamp"],
            "Damage types": ", ".join(
                sorted({d["class_name"] for d in detections_by_image.get(o["image_id"], [])})
            )
            or "none",
            "Risk score": o["image_risk_score"],
            "Category": o["risk_category"],
        }
        for o in recent
    ],
    width="stretch",
    hide_index=True,
)

# --- Overview map --------------------------------------------------------------------
st.subheader("Overview map")
pilot_area = load_app_config()["pilot_area"]
fmap = folium.Map(location=(pilot_area["center_lat"], pilot_area["center_lon"]), zoom_start=13)
for obs in observations:
    dets = detections_by_image.get(obs["image_id"], [])
    color = RISK_COLORS.get(obs["risk_category"], "#94A3B8") if dets else "#94A3B8"
    folium.CircleMarker(
        location=[obs["latitude"], obs["longitude"]],
        radius=7,
        color="#FFFFFF",
        weight=1.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.95,
        tooltip=obs["filename"],
    ).add_to(fmap)
st_folium(fmap, height=450, use_container_width=True, returned_objects=[])
