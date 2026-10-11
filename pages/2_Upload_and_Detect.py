"""Upload & Detect page.

Image upload -> GPS (EXIF or manual) -> YOLO detection -> review -> confirm
& save. See src/app/upload_pipeline.py for the underlying logic; this file
is intentionally just UI wiring around it.
"""

from __future__ import annotations

from PIL import Image

import streamlit as st

from src.app import database as db
from src.app import upload_pipeline as up
from src.app.resources import get_detector, get_graph
from src.app.sidebar import CONFIDENCE_KEY, render_sidebar
from src.app.theme import page_header, risk_badge_html
from src.detection.annotate import draw_annotations

st.set_page_config(page_title="Road Guard AI · Upload & Detect", page_icon="📤", layout="wide")
render_sidebar()
db.init_db()

page_header(
    "📤",
    "Upload & Detect",
    "Upload one or more road images, verify their location, run detection, and save confirmed observations.",
)

if "upload_state" not in st.session_state:
    st.session_state.upload_state = {}

uploaded_files = st.file_uploader(
    "Upload road images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Batch upload supported. Each image is checked for duplicates by file content.",
)

# Ingest any newly uploaded files (GPS extraction, duplicate/open checks) into session state.
if uploaded_files:
    for file in uploaded_files:
        file_bytes = file.getvalue()
        file_hash = up.compute_file_hash(file_bytes)
        if file_hash not in st.session_state.upload_state:
            processed = up.process_upload(file_bytes, file.name)
            st.session_state.upload_state[file_hash] = {
                "processed": processed,
                "latitude": processed.latitude,
                "longitude": processed.longitude,
                "coordinate_source": processed.coordinate_source,
                "detection_result": None,
                "saved": False,
            }

confidence_threshold = st.session_state.get(CONFIDENCE_KEY, 0.25)

pending_entries = [
    (h, s)
    for h, s in st.session_state.upload_state.items()
    if s["processed"].can_open and not s["processed"].already_saved and not s["saved"]
]

col_a, col_b = st.columns([1, 3])
with col_a:
    analyse_clicked = st.button("🔍 Analyse images", type="primary", disabled=not pending_entries)
with col_b:
    save_all_clicked = False
    ready_to_save = [
        (h, s)
        for h, s in pending_entries
        if s["detection_result"] is not None
        and up.validate_coordinates(s["latitude"] or 0, s["longitude"] or 0)
    ]
    if ready_to_save:
        save_all_clicked = st.button(f"💾 Save all valid observations ({len(ready_to_save)})")

if analyse_clicked:
    detector = get_detector()
    with st.spinner(f"Running detection on {len(pending_entries)} image(s)..."):
        for file_hash, state in pending_entries:
            state["detection_result"] = up.run_detection(detector, state["processed"].image_path)
    st.rerun()

if not st.session_state.upload_state:
    st.info("Upload at least one image to begin.", icon="ℹ️")

for file_hash, state in st.session_state.upload_state.items():
    processed = state["processed"]
    with st.expander(f"{processed.filename}", expanded=not state["saved"]):
        if processed.already_saved or state["saved"]:
            st.success("Already saved to the database.", icon="✅")
            continue

        if not processed.can_open:
            st.error(f"Could not open this file: {processed.open_error}", icon="⚠️")
            continue

        col_img, col_meta = st.columns([1, 1])
        with col_img:
            st.image(str(processed.image_path), caption="Original", width="stretch")

        with col_meta:
            if processed.gps is not None:
                st.success(
                    f"GPS from EXIF: {processed.gps.latitude:.6f}, {processed.gps.longitude:.6f}",
                    icon="📍",
                )
            else:
                st.warning(
                    "No GPS metadata found in this image. Enter coordinates manually.", icon="📍"
                )

            lat_key = f"lat_{file_hash}"
            lon_key = f"lon_{file_hash}"
            latitude = st.number_input(
                "Latitude",
                value=float(state["latitude"] or 0.0),
                min_value=-90.0,
                max_value=90.0,
                format="%.6f",
                key=lat_key,
            )
            longitude = st.number_input(
                "Longitude",
                value=float(state["longitude"] or 0.0),
                min_value=-180.0,
                max_value=180.0,
                format="%.6f",
                key=lon_key,
            )
            # If the user edited away from the original EXIF value, the source becomes manual.
            if processed.gps is not None and (
                abs(latitude - processed.gps.latitude) > 1e-6
                or abs(longitude - processed.gps.longitude) > 1e-6
            ):
                state["coordinate_source"] = "manual"
            state["latitude"], state["longitude"] = latitude, longitude

            if not up.validate_coordinates(latitude, longitude):
                st.error("Latitude must be in [-90, 90] and longitude in [-180, 180].", icon="⚠️")

            st.caption(f"Coordinate source: `{state['coordinate_source']}`")

        detection_result = state["detection_result"]
        if detection_result is None:
            st.caption("Not analysed yet -- click **Analyse images** above.")
            continue

        accepted = up.accepted_detections(detection_result.detections, confidence_threshold)
        st.divider()
        st.markdown(
            f"**Inference time:** {detection_result.inference_time_ms} ms &nbsp;|&nbsp; "
            f"**Accepted detections:** {len(accepted)} (threshold {confidence_threshold:.2f})"
        )

        if not accepted:
            st.info("No road damage detected above the current confidence threshold.", icon="✅")
        else:
            with Image.open(processed.image_path) as original_image:
                annotated = draw_annotations(original_image, accepted)
            col_orig, col_annot = st.columns(2)
            col_annot.image(annotated, caption="Annotated", width="stretch")

            st.dataframe(
                up.build_result_rows(accepted, detection_result.image_size),
                width="stretch",
                hide_index=True,
            )

            obs_risk = up.compute_observation_risk(accepted)
            st.markdown(
                f"**Observation risk:** {obs_risk.normalized_score:.2f} &nbsp; {risk_badge_html(obs_risk.category)}",
                unsafe_allow_html=True,
            )

        valid_coords = up.validate_coordinates(latitude, longitude)
        if st.button("💾 Save observation", key=f"save_{file_hash}", disabled=not valid_coords):
            match = up.save_observation(
                image_id=file_hash[:16],
                filename=processed.filename,
                file_hash=file_hash,
                image_path=processed.image_path,
                latitude=latitude,
                longitude=longitude,
                coordinate_source=state["coordinate_source"],
                confidence_threshold=confidence_threshold,
                inference_time_ms=detection_result.inference_time_ms,
                detections=accepted,
                graph=get_graph(),
            )
            state["saved"] = True
            if not match.in_scope:
                st.warning(
                    "Saved, but this location is outside the Sofia pilot area -- not matched to a road segment."
                )
            elif match.needs_manual_review:
                st.warning(
                    f"Saved, but the nearest road is {match.distance_m}m away -- flagged for manual review."
                )
            else:
                st.success(f"Saved and matched to a road segment ({match.distance_m}m away).")
            st.rerun()

if save_all_clicked:
    graph = get_graph()
    for file_hash, state in ready_to_save:
        accepted = up.accepted_detections(
            state["detection_result"].detections, confidence_threshold
        )
        up.save_observation(
            image_id=file_hash[:16],
            filename=state["processed"].filename,
            file_hash=file_hash,
            image_path=state["processed"].image_path,
            latitude=state["latitude"],
            longitude=state["longitude"],
            coordinate_source=state["coordinate_source"],
            confidence_threshold=confidence_threshold,
            inference_time_ms=state["detection_result"].inference_time_ms,
            detections=accepted,
            graph=graph,
        )
        state["saved"] = True
    st.success(f"Saved {len(ready_to_save)} observation(s).")
    st.rerun()
