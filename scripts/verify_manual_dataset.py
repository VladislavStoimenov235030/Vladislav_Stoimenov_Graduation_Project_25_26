"""Sanity-checks the manually collected/annotated dataset before it's merged
into any training split.

Checks, per image/label pair in data/manual/{images,labels}:
- every image has a matching label file and vice versa (no orphans)
- every label file is valid YOLO format (5 whitespace-separated fields per
  line, class_id in {0,1,2,3}, all coordinates normalized to [0, 1])
- the class(es) named in the filename (e.g. "D40_Pothole_manual_009.png")
  actually appear among that file's annotated classes -- catches an export
  with a shuffled/wrong class-id mapping early, before it corrupts training
- prints an overall class distribution, the same way the small/medium
  RDD2022 datasets were summarized when they were created

Read-only: never modifies data/manual, only reports.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

MANUAL_DIR = Path(__file__).resolve().parents[1] / "data" / "manual"
IMAGES_DIR = MANUAL_DIR / "images"
LABELS_DIR = MANUAL_DIR / "labels"

# Matches configs/datasets/rdd2022.yaml -- the single source of truth for
# which class code maps to which id everywhere else in this project.
CLASS_CODE_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
CLASS_ID_TO_NAME = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Returns (rows, errors) for one label file."""
    rows = []
    errors = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {lineno}: expected 5 fields, got {len(parts)} ({line!r})")
            continue
        try:
            class_id = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"line {lineno}: non-numeric field ({line!r})")
            continue

        if class_id not in CLASS_ID_TO_NAME:
            errors.append(f"line {lineno}: unknown class_id {class_id} (expected 0-3)")
        for name, value in (("x_center", x), ("y_center", y), ("width", w), ("height", h)):
            if not (0.0 <= value <= 1.0):
                errors.append(f"line {lineno}: {name}={value} out of [0, 1] range")

        rows.append((class_id, x, y, w, h))
    return rows, errors


def expected_class_id_from_filename(stem: str) -> int | None:
    for code, class_id in CLASS_CODE_TO_ID.items():
        if stem.startswith(code):
            return class_id
    return None


def main() -> None:
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise SystemExit(f"Expected {IMAGES_DIR} and {LABELS_DIR} to exist.")

    image_stems = {p.stem: p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    label_stems = {p.stem: p for p in LABELS_DIR.iterdir() if p.suffix.lower() == ".txt"}

    missing_labels = sorted(image_stems.keys() - label_stems.keys())
    missing_images = sorted(label_stems.keys() - image_stems.keys())

    class_counts: Counter[int] = Counter()
    empty_labels: list[str] = []
    format_errors: dict[str, list[str]] = {}
    filename_mismatches: list[str] = []

    for stem in sorted(image_stems.keys() & label_stems.keys()):
        rows, errors = parse_label_file(label_stems[stem])
        if errors:
            format_errors[stem] = errors
            continue
        if not rows:
            empty_labels.append(stem)
            continue

        found_classes = {row[0] for row in rows}
        class_counts.update(found_classes)

        expected = expected_class_id_from_filename(stem)
        if expected is not None and expected not in found_classes:
            filename_mismatches.append(
                f"{stem}: filename implies class {expected} ({CLASS_ID_TO_NAME[expected]}), "
                f"but label file only contains {sorted(found_classes)}"
            )

    print(f"Manual dataset: {MANUAL_DIR}")
    print(f"  images: {len(image_stems)}   labels: {len(label_stems)}")
    print()

    if missing_labels:
        print(f"MISSING LABEL FILES ({len(missing_labels)}):")
        for s in missing_labels:
            print(f"   {s}")
        print()
    if missing_images:
        print(f"MISSING IMAGE FILES ({len(missing_images)}):")
        for s in missing_images:
            print(f"   {s}")
        print()
    if format_errors:
        print(f"LABEL FORMAT ERRORS ({len(format_errors)} file(s)):")
        for stem, errors in format_errors.items():
            print(f"   {stem}.txt:")
            for e in errors:
                print(f"      {e}")
        print()
    if empty_labels:
        print(f"EMPTY LABEL FILES ({len(empty_labels)}) -- no boxes at all:")
        for s in empty_labels:
            print(f"   {s}")
        print()
    if filename_mismatches:
        print(f"FILENAME/CLASS MISMATCHES ({len(filename_mismatches)}) -- check these by hand:")
        for m in filename_mismatches:
            print(f"   {m}")
        print()

    print("Class distribution (images containing at least one box of that class):")
    for class_id, name in CLASS_ID_TO_NAME.items():
        print(f"   {class_id} {name:25s} {class_counts.get(class_id, 0)}")

    n_problems = (
        len(missing_labels) + len(missing_images) + len(format_errors) + len(filename_mismatches)
    )
    print()
    if n_problems == 0:
        print("All checks passed -- dataset looks ready to merge/split.")
    else:
        print(f"{n_problems} issue(s) found -- review before using this batch for training.")


if __name__ == "__main__":
    main()
