"""Cached, process-wide resources for the Streamlit app.

Spec requirement (Section 13): "Cache the loaded YOLO model" and "Cache the
OpenStreetMap graph" so they're loaded once per app process, not on every
page interaction. st.cache_resource is the correct Streamlit primitive for
this (unlike st.cache_data, it doesn't try to hash/copy the object).
"""

from __future__ import annotations

import streamlit as st

from src.config import load_app_config
from src.detection.detector import DamageDetector
from src.geospatial.network import get_sofia_graph


@st.cache_resource(show_spinner="Loading YOLO model...")
def get_detector() -> DamageDetector:
    """Load the model once, at the lowest confidence the UI allows.

    The confidence slider in the sidebar is meant to be adjustable without
    reloading the (cached) model, so inference runs at the permissive
    min_confidence_threshold and pages filter the returned detections by
    the user's chosen threshold afterward.
    """
    app_config = load_app_config()
    return DamageDetector(
        model_path=app_config["model"]["path"],
        confidence_threshold=app_config["model"]["min_confidence_threshold"],
    )


@st.cache_resource(show_spinner="Loading Sofia road network...")
def get_graph():
    return get_sofia_graph()
