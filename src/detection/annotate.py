"""Annotated-image rendering and bbox geometry helpers for the Upload &
Detect page (spec 2.4/2.6: annotated copy showing class + confidence next
to each box; relative bounding-box area displayed per detection).
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from src.detection.detector import Detection

BOX_COLOR = (255, 46, 46)
TEXT_BG = (255, 46, 46)
TEXT_COLOR = (255, 255, 255)


def relative_bbox_area(detection: Detection, image_width: int, image_height: int) -> float:
    """Bounding-box area as a fraction of the full image area, in [0, 1]."""
    x1, y1, x2, y2 = detection.bbox_xyxy
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = image_width * image_height
    return round(box_area / image_area, 4) if image_area > 0 else 0.0


def draw_annotations(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Return a copy of `image` with bounding boxes + "class conf" labels drawn on it."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=max(2, image.width // 400))

        label = f"{det.class_name} {det.confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label)
        text_h = text_bbox[3] - text_bbox[1]
        label_y = y1 - text_h - 6 if y1 - text_h - 6 > 0 else y1 + 2
        draw.rectangle(
            [x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 8, label_y + text_h + 6],
            fill=TEXT_BG,
        )
        draw.text((x1 + 4, label_y + 2), label, fill=TEXT_COLOR)

    return annotated
