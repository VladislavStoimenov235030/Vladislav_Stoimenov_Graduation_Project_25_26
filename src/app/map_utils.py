"""Map-building helpers for the Damage Map page (spec Section 4).

Kept separate from the page so the map construction logic (marker popups,
road-segment coloring, heatmap) can be reasoned about and reused
independently of Streamlit widget wiring.
"""

from __future__ import annotations

import base64
import io
import sqlite3
from pathlib import Path

import folium
from folium.plugins import HeatMap
from PIL import Image

from src.app.theme import RISK_COLORS, risk_badge_html
from src.detection.annotate import draw_annotations
from src.detection.detector import Detection

# folium.Marker/Icon only accepts a fixed set of named colors, not arbitrary
# hex -- these are the closest named matches to the theme's risk palette.
CATEGORY_COLOR = {"low": "green", "medium": "orange", "high": "red"}
NO_DAMAGE_COLOR = "gray"
POPUP_THUMBNAIL_WIDTH = 380


def _rows_to_detections(image_path: str, detections: list[sqlite3.Row]) -> list[Detection]:
    return [
        Detection(
            image_path=image_path,
            class_id=d["class_id"],
            class_name=d["class_name"],
            class_code=d["class_code"],
            confidence=d["confidence"],
            bbox_xyxy=(d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]),
        )
        for d in detections
    ]


def _thumbnail_data_uri(image_path: str, detections: list[sqlite3.Row]) -> str | None:
    """Annotated (bounding boxes drawn) thumbnail, so the popup shows what the model found."""
    path = Path(image_path)
    if not path.exists():
        return None
    with Image.open(path) as img:
        if detections:
            img = draw_annotations(img, _rows_to_detections(image_path, detections))
        else:
            img = img.convert("RGB")
        ratio = POPUP_THUMBNAIL_WIDTH / img.width
        img = img.resize((POPUP_THUMBNAIL_WIDTH, int(img.height * ratio)))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def build_popup_html(observation: sqlite3.Row, detections: list[sqlite3.Row]) -> str:
    """Spec 4.1: image preview, timestamp, coordinates, source, damage types,
    detection count, highest confidence, risk score/category, road segment.

    The image is annotated with the model's bounding boxes (see
    _thumbnail_data_uri) so a click on the map shows exactly what was
    detected, not just the raw photo.
    """
    thumb = _thumbnail_data_uri(observation["image_path"], detections)
    img_html = (
        f'<img src="{thumb}" width="{POPUP_THUMBNAIL_WIDTH}" '
        f'style="border-radius:10px;display:block;margin-bottom:10px;">'
        if thumb
        else ""
    )

    damage_types = sorted({d["class_name"] for d in detections}) or ["None"]
    max_conf = max((d["confidence"] for d in detections), default=None)
    max_conf_str = f"{max_conf:.2f}" if max_conf is not None else "n/a"

    if observation["matched_edge_u"] is not None:
        segment = f"{observation['matched_edge_u']}_{observation['matched_edge_v']}_{observation['matched_edge_key']}"
    else:
        segment = "not matched (outside pilot area)"

    badge = (
        risk_badge_html(observation["risk_category"])
        if detections
        else ('<span class="risk-badge" style="background:#94A3B8;">No damage</span>')
    )
    row = lambda label, value: (  # noqa: E731 -- tiny local formatter, not worth a def
        f'<div style="display:flex;justify-content:space-between;gap:12px;'
        f'padding:3px 0;border-bottom:1px solid #EEF1F6;">'
        f'<span style="color:#64748B;">{label}</span>'
        f'<span style="color:#0F172A;font-weight:500;text-align:right;">{value}</span></div>'
    )

    return f"""
    <div style="font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; font-size: 13px; width: {POPUP_THUMBNAIL_WIDTH}px;">
        <style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');</style>
        {img_html}
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <b style="font-size:14px;color:#0B1E3D;">{observation['filename']}</b>
            {badge}
        </div>
        {row("Processed", observation['upload_timestamp'])}
        {row("Location", f"{observation['latitude']:.6f}, {observation['longitude']:.6f} ({observation['coordinate_source']})")}
        {row("Damage types", ', '.join(damage_types))}
        {row("Detections", len(detections))}
        {row("Highest confidence", max_conf_str)}
        {row("Risk score", f"{observation['image_risk_score']:.2f}")}
        {row("Road segment", segment)}
    </div>
    """


def add_observation_markers(
    fmap: folium.Map, observations: list[sqlite3.Row], detections_by_image: dict
) -> None:
    layer = folium.FeatureGroup(name="Observations", show=True)
    for obs in observations:
        dets = detections_by_image.get(obs["image_id"], [])
        color = (
            CATEGORY_COLOR.get(obs["risk_category"], NO_DAMAGE_COLOR) if dets else NO_DAMAGE_COLOR
        )
        folium.Marker(
            location=[obs["latitude"], obs["longitude"]],
            # max_height + Leaflet's own autoPan (default on) together keep the
            # whole popup on-screen inside the map area, instead of being
            # clipped below the visible map and needing a page scroll.
            popup=folium.Popup(
                build_popup_html(obs, dets),
                max_width=POPUP_THUMBNAIL_WIDTH + 30,
                max_height=520,
            ),
            tooltip=obs["filename"],
            icon=folium.Icon(color=color, icon="road", prefix="fa"),
        ).add_to(layer)
    layer.add_to(fmap)


def _edge_coords(graph, u: int, v: int, key: int) -> list[tuple[float, float]]:
    edge_data = graph.get_edge_data(u, v, key)
    geometry = edge_data.get("geometry")
    if geometry is not None:
        return [(lat, lon) for lon, lat in geometry.coords]
    return [(graph.nodes[u]["y"], graph.nodes[u]["x"]), (graph.nodes[v]["y"], graph.nodes[v]["x"])]


def add_segment_layer(fmap: folium.Map, segments: list[sqlite3.Row], graph) -> None:
    """Spec 4.3: color matched road segments by aggregated risk."""
    layer = folium.FeatureGroup(name="Road-segment risk", show=True)
    for seg in segments:
        try:
            coords = _edge_coords(graph, seg["edge_u"], seg["edge_v"], seg["edge_key"])
        except Exception:
            continue
        color = CATEGORY_COLOR.get(seg["risk_category"], "blue")
        folium.PolyLine(
            coords,
            color=color,
            weight=6,
            opacity=0.8,
            tooltip=f"{seg['road_name'] or 'unnamed road'} -- risk {seg['aggregated_risk']:.2f} "
            f"({seg['risk_category']}, {seg['observation_count']} observation(s))",
        ).add_to(layer)
    layer.add_to(fmap)


def add_heatmap_layer(fmap: folium.Map, observations: list[sqlite3.Row]) -> None:
    """Spec 4.4: weighted risk heatmap from GPS coordinates and normalized risk scores."""
    points = [
        [obs["latitude"], obs["longitude"], max(obs["image_risk_score"], 0.05)]
        for obs in observations
    ]
    if not points:
        return
    layer = folium.FeatureGroup(name="Risk heatmap", show=False)
    HeatMap(points, radius=20, blur=15).add_to(layer)
    layer.add_to(fmap)


def build_damage_map(
    observations: list[sqlite3.Row],
    detections_by_image: dict,
    segments: list[sqlite3.Row],
    graph,
    center: tuple[float, float],
) -> folium.Map:
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    add_segment_layer(fmap, segments, graph)
    add_observation_markers(fmap, observations, detections_by_image)
    add_heatmap_layer(fmap, observations)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
