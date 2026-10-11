"""Analyses the qualitative usability survey responses.

Input: thesis/qualitative_survey_responses_template.csv, filled in with
one row per participant (e.g. exported directly from Google Forms and
renamed/re-saved to match this template's column names).

Produces:
- Descriptive statistics (mean, median, min, max) for the five closed
  (Likert 1-5) questions, plus a bar chart of the means.
- All open-ended responses printed grouped by question, ready for manual
  thematic coding (this script does not attempt automated theme
  extraction -- reading and grouping the actual responses into themes is
  the qualitative analysis step itself, done by the researcher).

Read-only: never modifies the input CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESPONSES_CSV = (
    Path(__file__).resolve().parents[1] / "thesis" / "qualitative_survey_responses_template.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"

LIKERT_QUESTIONS = {
    "q1_detection_easy": "Detection results were easy to understand",
    "q2_risk_intuitive": "Risk score/category was intuitive",
    "q3_map_clear": "Map made clear which roads were risky",
    "q4_trust": "Would trust the system's risk assessment",
    "q5_route_useful": "Route comparison was useful",
}

OPEN_QUESTIONS = {
    "q6_confusing_open": "What was confusing or unclear?",
    "q7_trust_reason_open": "Would you trust this system? Why or why not?",
    "q8_improvement_open": "What would make this more useful?",
    "q9_other_open": "Any other comments?",
}


def main() -> None:
    if not RESPONSES_CSV.exists():
        raise FileNotFoundError(f"{RESPONSES_CSV} not found.")

    df = pd.read_csv(RESPONSES_CSV)
    df = df.dropna(how="all")

    if df.empty:
        print(
            f"{RESPONSES_CSV} has no responses yet -- fill it in first "
            "(or export from Google Forms and match the column names)."
        )
        return

    print(f"{len(df)} response(s) loaded.\n")

    # --- Closed (Likert) questions: descriptive statistics ------------------
    print("=== Closed questions (1-5 scale) ===\n")
    stats_rows = []
    for col, label in LIKERT_QUESTIONS.items():
        if col not in df.columns:
            print(f"Warning: column '{col}' not found, skipping")
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        stats_rows.append(
            {
                "question": label,
                "n": len(values),
                "mean": round(values.mean(), 2),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False))

    if not stats_df.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stats_path = OUTPUT_DIR / "qualitative_survey_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nSaved to {stats_path}")

        plt.figure(figsize=(8, 4))
        plt.barh(stats_df["question"], stats_df["mean"])
        plt.xlim(1, 5)
        plt.xlabel("Mean rating (1-5)")
        plt.title("Usability survey -- mean ratings")
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "qualitative_survey_means.png"
        plt.savefig(fig_path)
        print(f"Chart saved to {fig_path}")

    # --- Open-ended questions: grouped listing for manual thematic coding --
    print("\n\n=== Open-ended responses (group into themes manually) ===")
    for col, label in OPEN_QUESTIONS.items():
        if col not in df.columns:
            continue
        print(f"\n--- {label} ---")
        for i, response in enumerate(df[col].dropna(), start=1):
            response = str(response).strip()
            if response:
                print(f"  [{i}] {response}")


if __name__ == "__main__":
    main()
