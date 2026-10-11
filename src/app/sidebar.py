"""Shared sidebar, rendered identically on every page.

Spec requirement (Section "Application pages"): the sidebar must show page
navigation (Streamlit provides this automatically for files under pages/),
loaded model name/version, model status, stored image/detection counts,
the configured confidence threshold, and a button to clear active filters.
"""

from __future__ import annotations

import streamlit as st

from src.app import database as db
from src.app.theme import inject_custom_css
from src.config import load_app_config

CONFIDENCE_KEY = "confidence_threshold"


def render_sidebar() -> None:
    inject_custom_css()

    app_config = load_app_config()
    model_cfg = app_config["model"]

    st.sidebar.markdown("### 🛡️ Road Guard AI")
    st.sidebar.caption("Municipality road-risk decision-support prototype")

    st.sidebar.divider()
    st.sidebar.markdown("**Model**")
    st.sidebar.text(model_cfg["version_label"])
    from pathlib import Path

    model_path = Path(__file__).resolve().parents[2] / model_cfg["path"]
    if model_path.exists():
        st.sidebar.success("Model available", icon="✅")
    else:
        st.sidebar.error("Model file not found", icon="⚠️")

    st.sidebar.divider()
    st.sidebar.markdown("**Stored data**")
    try:
        db.init_db()
        n_obs = db.count_observations()
        n_det = db.count_detections()
    except Exception:
        n_obs, n_det = 0, 0
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Images", n_obs)
    col2.metric("Detections", n_det)

    st.sidebar.divider()
    st.sidebar.markdown("**Detection settings**")
    if CONFIDENCE_KEY not in st.session_state:
        st.session_state[CONFIDENCE_KEY] = model_cfg["default_confidence_threshold"]
    st.sidebar.slider(
        "Confidence threshold",
        min_value=model_cfg["min_confidence_threshold"],
        max_value=model_cfg["max_confidence_threshold"],
        step=0.05,
        key=CONFIDENCE_KEY,
        help="Detections below this confidence are discarded before risk scoring.",
    )

    st.sidebar.divider()
    if st.sidebar.button("Clear active filters", width="stretch"):
        for key in list(st.session_state.keys()):
            if key.startswith("filter_"):
                del st.session_state[key]
        st.sidebar.success("Filters cleared.")
