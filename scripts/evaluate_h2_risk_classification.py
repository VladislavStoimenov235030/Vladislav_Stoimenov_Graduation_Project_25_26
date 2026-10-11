"""H2 evaluation: rule-based risk category vs. manually assigned ground truth.

H2: "A rule-based risk scoring model based on damage type and size will
effectively differentiate road segments into low, medium, and high-risk
categories." Operationalized (Streamlit_application_instructions.md 8.3) as:
accuracy, precision, recall, F1, confusion matrix, target F1 >= 0.75.

Input: thesis/h2_manual_risk_labels_template.csv, filled in with a
'manual_risk_label' (low/medium/high) for each of the 42 held-out Sofia
test images -- assigned by looking at the photo alone, blind to the
model's output (see conversation/thesis notes for why this order matters).

For each image: runs the actual deployed detection model
(src.detection.detector.DamageDetector, i.e. whatever config/app_config.yaml
currently points to) + the actual risk-scoring pipeline
(src.risk.scoring + src.risk.observation_risk) to get the category the app
would really show a user -- not a re-derivation, the real pipeline.

Also reports Cohen's kappa (chance-corrected agreement, with its own
significance test) alongside the spec's required metrics, since a plain
accuracy/F1 number does not account for how much agreement would be
expected by chance alone with 3 categories.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.detector import DamageDetector
from src.risk.observation_risk import observation_risk_score
from src.risk.scoring import detection_risk_score

LABELS_CSV = Path(__file__).resolve().parents[1] / "thesis" / "h2_manual_risk_labels_template.csv"
IMAGES_DIR = Path(__file__).resolve().parents[1] / "data" / "manual_split" / "test" / "images"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"

CATEGORIES = ["low", "medium", "high"]


def load_manual_labels(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["manual_risk_label"].strip().lower()
            if not label:
                continue
            if label not in CATEGORIES:
                raise ValueError(
                    f"{row['image_filename']}: manual_risk_label={label!r} is not one of {CATEGORIES}"
                )
            rows.append({"image_filename": row["image_filename"], "manual_label": label})
    return rows


def predicted_category(detector: DamageDetector, image_path: Path) -> str:
    """Runs the real detection + risk-scoring pipeline, same as the app does."""
    detections = detector.predict(image_path)
    accepted = [d for d in detections if d.confidence >= detector.confidence_threshold]
    if not accepted:
        return "low"  # matches upload_pipeline.py: no detections -> category "low", risk 0.0

    risks = [detection_risk_score(d.class_name, d.confidence) for d in accepted]
    result = observation_risk_score(risks)
    return result.category if result else "low"


def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"{LABELS_CSV} not found. Run this after filling in thesis/h2_manual_risk_labels_template.csv"
        )

    manual_rows = load_manual_labels(LABELS_CSV)
    if len(manual_rows) < len(list(IMAGES_DIR.iterdir())):
        print(
            f"Note: {len(manual_rows)} of {len(list(IMAGES_DIR.iterdir()))} images have a manual "
            "label filled in -- evaluating on the labeled subset only."
        )
    if not manual_rows:
        raise ValueError("No manual labels found -- fill in the manual_risk_label column first.")

    print("Loading detector (whatever config/app_config.yaml currently points to)...")
    detector = DamageDetector()
    print(f"Model: {detector.model_path}")
    print(f"Confidence threshold: {detector.confidence_threshold}\n")

    results = []
    for row in manual_rows:
        image_path = IMAGES_DIR / row["image_filename"]
        if not image_path.exists():
            print(f"Warning: {image_path} not found, skipping")
            continue
        pred = predicted_category(detector, image_path)
        results.append(
            {
                "image_filename": row["image_filename"],
                "manual": row["manual_label"],
                "predicted": pred,
            }
        )

    results_df = pd.DataFrame(results)
    print(f"Evaluated {len(results_df)} images.\n")

    y_true = results_df["manual"]
    y_pred = results_df["predicted"]

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=CATEGORIES)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Cohen's kappa (chance-corrected agreement): {kappa:.3f}")
    print()
    print("Classification report (per-category precision/recall/F1):")
    print(classification_report(y_true, y_pred, labels=CATEGORIES, zero_division=0))

    macro_f1 = classification_report(
        y_true, y_pred, labels=CATEGORIES, output_dict=True, zero_division=0
    )["macro avg"]["f1-score"]
    print(
        f"H2 target (macro F1 >= 0.75): {'SUPPORTED' if macro_f1 >= 0.75 else 'NOT SUPPORTED'} (macro F1 = {macro_f1:.3f})"
    )

    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES).plot(ax=ax, cmap="Blues")
    ax.set_title("H2: Manual risk label vs. rule-based prediction")
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "h2_confusion_matrix.png"
    plt.savefig(fig_path)
    print(f"\nConfusion matrix saved to {fig_path}")

    results_path = OUTPUT_DIR / "h2_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Per-image results saved to {results_path}")


if __name__ == "__main__":
    main()
