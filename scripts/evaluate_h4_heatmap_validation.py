"""H4 evaluation: generated segment risk vs. manually assigned ground truth.

H4: "The system will enable identification of high-risk road segments..."
Operationalized (Streamlit_application_instructions.md 8.5) as: Spearman
correlation between generated segment/location risk and manually labeled
pilot-area risk, target rho >= 0.5.

Input: thesis/h4_manual_segment_ratings_template.csv, filled in with a
'manual_risk_rating_1to5' (integer 1-5) for each of the 16 real Sveta
Troitsa road segments -- assigned by independently judging each street
(via the Google Maps link, personal knowledge, or a visit), blind to the
model's computed risk score.

Compares against the segment's live aggregated_risk and risk_density
(src.app.database.get_all_road_segments()) -- the same numbers the app's
Damage Map page displays -- using scipy.stats.spearmanr, which reports
both rho and its own significance p-value.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

# Road names are Cyrillic; Windows' default console codepage (cp1252) can't
# encode them, which would otherwise crash a plain `python scripts/...` run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import database as db

LABELS_CSV = (
    Path(__file__).resolve().parents[1] / "thesis" / "h4_manual_segment_ratings_template.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"

RHO_TARGET = 0.5


def load_manual_ratings(csv_path: Path) -> dict[str, int]:
    ratings = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row["manual_risk_rating_1to5"].strip()
            if not value:
                continue
            rating = int(value)
            if not (1 <= rating <= 5):
                raise ValueError(f"{row['segment_id']}: rating {rating} is not in 1-5")
            ratings[row["segment_id"]] = rating
    return ratings


def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"{LABELS_CSV} not found. Run this after filling in "
            "thesis/h4_manual_segment_ratings_template.csv"
        )

    ratings = load_manual_ratings(LABELS_CSV)
    if not ratings:
        raise ValueError("No manual ratings found -- fill in manual_risk_rating_1to5 first.")

    db.init_db()
    segments = {
        f"{s['edge_u']}_{s['edge_v']}_{s['edge_key']}": s for s in db.get_all_road_segments()
    }

    rows = []
    for segment_id, rating in ratings.items():
        seg = segments.get(segment_id)
        if seg is None:
            print(f"Warning: {segment_id} not found in the live database, skipping")
            continue
        rows.append(
            {
                "segment_id": segment_id,
                "road_name": seg["road_name"] or "(unnamed)",
                "manual_rating": rating,
                "aggregated_risk": seg["aggregated_risk"],
                "risk_density": seg["risk_density"],
            }
        )

    if len(rows) < 4:
        print(
            f"Only {len(rows)} rated segments matched -- Spearman's test needs a reasonable "
            "sample size to be meaningful; proceeding, but treat the result as preliminary."
        )

    df = pd.DataFrame(rows)
    print(f"Evaluated {len(df)} rated segments.\n")
    print(df.to_string(index=False))
    print()

    for risk_column in ["aggregated_risk", "risk_density"]:
        rho, p_value = stats.spearmanr(df["manual_rating"], df[risk_column])
        print(
            f"Spearman correlation (manual rating vs. {risk_column}): "
            f"rho={rho:.3f}, p={p_value:.4f}"
        )
        verdict = "SUPPORTED" if rho >= RHO_TARGET and p_value < 0.05 else "NOT SUPPORTED"
        print(f"  H4 target (rho >= {RHO_TARGET}, p < 0.05): {verdict}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "h4_evaluation_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Per-segment results saved to {results_path}")


if __name__ == "__main__":
    main()
