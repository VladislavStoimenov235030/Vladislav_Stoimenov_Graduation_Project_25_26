"""Damage detection module.

Thin wrapper around the trained YOLOv8n model so the rest of the pipeline
(risk scoring, geospatial mapping, Streamlit) can call one simple function
and get back plain Python objects instead of dealing with ultralytics
Results objects directly.

Model used: YOLOv8n trained on the class/country-balanced RDD2022 "large"
4-class dataset (notebooks/09_yolov8n_large_dataset.ipynb, 4,291 train
images), then fine-tuned on 131 real Sofia photos for domain adaptation
(notebooks/11_yolov8n_sofia_finetune.ipynb).
Known performance on the RDD2022-large test set (pre-fine-tune): mAP50
0.548, mAP50-95 0.300. Known performance on the held-out Sofia test set
(post-fine-tune, the number that matters for this app's actual use case):
mAP50 0.278, mAP50-95 0.148, F1 0.347 (vs. 0.189 before fine-tuning).
See thesis/model_training_results_and_discussion.md for the full comparison
and the reasoning for using the fine-tuned model here despite its lower
RDD2022-benchmark score.

The model's raw class names (baked into the checkpoint from the training
data.yaml, e.g. "D00_Longitudinal_Crack") are translated here to a
professional display name (config/app_config.yaml:
damage_class_display_names) before they ever leave this module -- every
downstream consumer (risk scoring, annotated images, map popups, database)
works with the clean name, not the raw dataset code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from src.config import load_app_config

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "runs"
    / "detect"
    / "runs"
    / "road_damage_sofia_finetune"
    / "yolov8n_sofia_finetune"
    / "weights"
    / "best.pt"
)


@dataclass
class Detection:
    """One detected damage instance in one image."""

    image_path: str
    class_id: int
    class_name: str  # professional display name, e.g. "Longitudinal Crack"
    class_code: str  # official RDD2022 code, e.g. "D00" -- for compact/technical use
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in pixels


class DamageDetector:
    """Loads the YOLO model once, then runs predictions on demand."""

    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL_PATH, confidence_threshold: float = 0.25
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {self.model_path}")
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(str(self.model_path))

    def predict(self, image_path: str | Path) -> list[Detection]:
        """Run detection on a single image, returning structured Detection records."""
        image_path = Path(image_path)
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            verbose=False,
        )

        display_names = load_app_config()["damage_class_display_names"]

        detections: list[Detection] = []
        result = results[0]
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            xmin, ymin, xmax, ymax = (float(v) for v in box.xyxy[0].tolist())
            raw_name = names[class_id]
            detections.append(
                Detection(
                    image_path=str(image_path),
                    class_id=class_id,
                    class_name=display_names.get(raw_name, raw_name),
                    class_code=raw_name.split("_")[0],
                    confidence=confidence,
                    bbox_xyxy=(xmin, ymin, xmax, ymax),
                )
            )
        return detections


if __name__ == "__main__":
    # Quick manual check against the real geotagged test images.
    test_dir = Path(__file__).resolve().parents[2] / "data" / "pipeline_test"
    detector = DamageDetector()
    for path in sorted(test_dir.glob("*.jpg")):
        dets = detector.predict(path)
        print(f"{path.name}: {len(dets)} detection(s)")
        for d in dets:
            print(f"   {d.class_name} conf={d.confidence:.2f} bbox={d.bbox_xyxy}")
