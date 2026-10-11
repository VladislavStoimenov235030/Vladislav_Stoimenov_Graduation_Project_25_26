"""Road Guard AI -- Streamlit entry point (landing page).

The actual workflow lives in pages/ (Streamlit's built-in multipage
mechanism, auto-discovered from the pages/ directory): 1_Dashboard,
2_Upload_and_Detect, 3_Damage_Map, 4_Safer_Route_Planner.
"""

from pathlib import Path

import streamlit as st

from src.app import database as db
from src.app.sidebar import render_sidebar
from src.app.theme import stat_card_html, step_card_html
from src.config import load_app_config

st.set_page_config(page_title="Road Guard AI", page_icon="🛡️", layout="wide")

render_sidebar()

HERO_IMAGE = Path(__file__).resolve().parent / "app_theme_photo.png"
if HERO_IMAGE.exists():
    st.image(str(HERO_IMAGE), width="stretch")

st.markdown(
    """
    <h2 style="margin-top:1.4rem;margin-bottom:0.3rem;">
        AI-powered road-risk mapping and <span style="color:#2563EB;">safer route planning</span>
    </h2>
    <p style="color:#64748B;font-size:1.02rem;max-width:70ch;">
        Detect road damage, map hazardous segments, and find safer routes.
    </p>
    """,
    unsafe_allow_html=True,
)

cta_col1, cta_col2, _ = st.columns([1.3, 1.3, 3])
with cta_col1:
    if st.button("Upload Images", type="primary", width="stretch"):
        st.switch_page("pages/2_Upload_and_Detect.py")
with cta_col2:
    if st.button("Explore Map", width="stretch"):
        st.switch_page("pages/3_Damage_Map.py")

st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# --- How it works: 4-step process strip --------------------------------------------------
steps = [
    (1, "Upload image", "Upload road photos or enter GPS coordinates."),
    (2, "Detect damage", "AI detects and classifies road damage."),
    (3, "Map road risk", "Damage data is mapped to road segments."),
    (4, "Find safer route", "Get a safer alternative with minimal extra distance."),
]
step_cols = st.columns([3, 0.4, 3, 0.4, 3, 0.4, 3])
for i, (number, title, desc) in enumerate(steps):
    step_cols[i * 2].markdown(step_card_html(number, title, desc), unsafe_allow_html=True)
    if i < len(steps) - 1:
        step_cols[i * 2 + 1].markdown('<div class="step-chevron">›</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

# --- Summary stat cards --------------------------------------------------------------------
db.init_db()
n_obs = db.count_observations()
n_det = db.count_detections()
n_high_risk_segments = len([s for s in db.get_all_road_segments() if s["risk_category"] == "high"])

model_cfg = load_app_config()["model"]
model_path = Path(__file__).resolve().parent / model_cfg["path"]
model_ok = model_path.exists()

s1, s2, s3, s4 = st.columns(4)
s1.markdown(
    stat_card_html(str(n_obs), "Images", "Total uploaded", "#2563EB"), unsafe_allow_html=True
)
s2.markdown(
    stat_card_html(str(n_det), "Detections", "Total detected", "#16A34A"), unsafe_allow_html=True
)
s3.markdown(
    stat_card_html(str(n_high_risk_segments), "Risk Locations", "High-risk segments", "#F59E0B"),
    unsafe_allow_html=True,
)
s4.markdown(
    stat_card_html(
        "Available" if model_ok else "Missing",
        model_cfg["version_label"],
        "Ready" if model_ok else "Check path",
        "#16A34A" if model_ok else "#DC2626",
    ),
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

if n_obs == 0:
    st.info(
        "No observations saved yet. Start on the **Upload & Detect** page to add "
        "geolocated road images before the map and route planner have any data to use.",
        icon="ℹ️",
    )

st.markdown(
    """
> Road image → GPS location → damage detection → risk score →
> OpenStreetMap road segment → risk map → safer route → measurable route comparison
"""
)
