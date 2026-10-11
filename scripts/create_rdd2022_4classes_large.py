"""Builds the "large" processed RDD2022 dataset -- the final-iteration
training set, meant to fix the two imbalances found in the small/medium
datasets (see thesis notes from the dataset-audit conversation):

1. Class imbalance: D00 (longitudinal crack) outnumbered D20/D40 by 2-3x in
   every previous processed split, even though D20/D40 are the highest-
   severity classes in the risk-scoring model (src/risk/scoring.py). This
   script samples close to an equal per-class target instead.

2. Country/source imbalance: Japan was ~46% of both previous processed
   datasets vs. ~28% of the raw pool; Czech Republic was barely represented
   at all (0 images in the small dataset's val split). This script splits
   each class's target evenly across whichever countries actually contain
   that class ("fair-share" allocation, see fair_share_sample()), instead
   of plain random sampling that lets the largest country dominate.

Important constraint discovered while auditing raw data availability:
D40 (pothole) has ZERO images in Norway, United_States, and Czech in this
dataset -- country-balancing is therefore done per-class using only the
countries that actually contain that class, not a fixed 7-country target.

Same processing rules as create_rdd2022_4classes_medium.py: raw data is
read-only, class 4 (and anything outside 0-3) is filtered out of copied
labels, and negative (no-damage) images are added as background examples.
"""

from pathlib import Path
import re
import shutil
import random
import yaml


RAW_ROOT = Path("data/raw/RDD2022")
OUT_ROOT = Path("data/processed/RDD2022_4classes_large")

VALID_CLASSES = {0, 1, 2, 3}

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}

COUNTRIES = ["Japan", "Norway", "India", "United_States", "Czech", "China_Drone", "China_MotorBike"]
COUNTRY_PATTERN = re.compile(
    r"^(China_Drone|China_MotorBike|Czech|India|Japan|Norway|United_States)_"
)

# Per-class positive samples (an image can satisfy more than one class, so
# unique image counts per split end up somewhat lower than 4x this number).
SPLIT_TARGETS = {
    "train": 1100,
    "val": 150,
    "test": 150,
}

# No-damage images added so the model also sees clean road as background
# (same rationale as the medium dataset).
NEGATIVE_TARGETS = {
    "train": 200,
    "val": 30,
    "test": 30,
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def find_image(image_dir: Path, image_id: str):
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{image_id}{ext}"
        if image_path.exists():
            return image_path
    return None


def read_label(label_path: Path):
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    return [line for line in lines if line.strip()]


def get_classes_from_label(label_path: Path):
    classes = []

    for line in read_label(label_path):
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        classes.append(class_id)

    return classes


def country_of(image_id: str) -> str:
    match = COUNTRY_PATTERN.match(image_id)
    return match.group(1) if match else "Unknown"


def filter_label_to_4_classes(raw_label_path: Path, output_label_path: Path):
    kept_lines = []

    for line in read_label(raw_label_path):
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))

        if class_id in VALID_CLASSES:
            kept_lines.append(line)

    output_label_path.write_text("\n".join(kept_lines), encoding="utf-8")


def fair_share_sample(candidates_by_country: dict, n_target: int) -> list:
    """Water-filling allocation: split n_target as evenly as possible across
    every country that has at least one candidate, redistributing any
    shortfall (a country running out of images) to the remaining countries
    instead of just taking whatever a plain random sample would give (which
    would let the largest country dominate, as happened with Japan in the
    small/medium datasets).
    """
    pools = {country: list(ids) for country, ids in candidates_by_country.items() if ids}
    selected = []
    remaining_target = n_target

    while remaining_target > 0 and pools:
        share = max(1, remaining_target // len(pools))
        made_progress = False

        for country in list(pools.keys()):
            if remaining_target <= 0:
                break

            take = min(share, len(pools[country]), remaining_target)
            if take <= 0:
                continue

            chosen = random.sample(pools[country], take)
            selected.extend(chosen)
            for image_id in chosen:
                pools[country].remove(image_id)

            remaining_target -= take
            made_progress = True

            if not pools[country]:
                del pools[country]

        if not made_progress:
            break

    return selected


def sample_images_for_class(split: str, class_id: int, n_samples: int) -> tuple:
    """Returns (selected_ids, per_country_breakdown) for one class, balanced
    across countries via fair_share_sample()."""
    label_dir = RAW_ROOT / split / "labels"

    candidates_by_country = {country: [] for country in COUNTRIES}
    for label_path in label_dir.glob("*.txt"):
        if class_id in get_classes_from_label(label_path):
            candidates_by_country[country_of(label_path.stem)].append(label_path.stem)

    total_available = sum(len(ids) for ids in candidates_by_country.values())
    if total_available < n_samples:
        print(
            f"Warning: split={split}, class={class_id} has only {total_available} images total. Requested {n_samples}."
        )

    selected = fair_share_sample(candidates_by_country, n_samples)

    breakdown = {
        country: sum(1 for image_id in selected if country_of(image_id) == country)
        for country in COUNTRIES
    }
    return selected, breakdown


def sample_negative_images(split: str, n_samples: int, exclude_ids: set):
    label_dir = RAW_ROOT / split / "labels"

    candidate_image_ids = []

    for label_path in label_dir.glob("*.txt"):
        if label_path.stem in exclude_ids:
            continue

        classes = set(get_classes_from_label(label_path))

        if not (classes & VALID_CLASSES):
            candidate_image_ids.append(label_path.stem)

    if len(candidate_image_ids) < n_samples:
        print(
            f"Warning: split={split} has only {len(candidate_image_ids)} no-damage images. Requested {n_samples}."
        )

    n_samples = min(n_samples, len(candidate_image_ids))

    return random.sample(candidate_image_ids, n_samples)


def copy_image_and_label(split: str, image_id: str):
    raw_image_dir = RAW_ROOT / split / "images"
    raw_label_dir = RAW_ROOT / split / "labels"

    out_image_dir = OUT_ROOT / split / "images"
    out_label_dir = OUT_ROOT / split / "labels"

    image_path = find_image(raw_image_dir, image_id)
    label_path = raw_label_dir / f"{image_id}.txt"

    if image_path is None:
        print(f"Image not found: {split}/{image_id}")
        return

    if not label_path.exists():
        print(f"Label not found: {split}/{image_id}")
        return

    out_image_path = out_image_dir / image_path.name
    out_label_path = out_label_dir / label_path.name

    shutil.copy2(image_path, out_image_path)
    filter_label_to_4_classes(label_path, out_label_path)


def create_folders():
    for split in SPLIT_TARGETS:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def count_output():
    print("\nCreated large dataset:")

    for split in SPLIT_TARGETS:
        image_dir = OUT_ROOT / split / "images"
        label_dir = OUT_ROOT / split / "labels"

        images = list(image_dir.glob("*"))
        labels = list(label_dir.glob("*.txt"))

        negative_labels = sum(
            1 for label_path in label_dir.glob("*.txt") if len(read_label(label_path)) == 0
        )

        country_counts = {country: 0 for country in COUNTRIES}
        for image_path in images:
            country_counts[country_of(image_path.stem)] += 1

        print(
            f"\n{split}: {len(images)} images, {len(labels)} labels ({negative_labels} with no damage box)"
        )
        for country, n in country_counts.items():
            if n:
                print(f"   {country:18s} {n}")


def write_data_yaml():
    # Ultralytics resolves a *relative* `path:` against its own global
    # `datasets_dir` setting, not against the yaml file or cwd -- so a
    # relative path here silently breaks dataset loading. An absolute path,
    # recomputed by this script on every machine it runs on, is the correct
    # approach; data.yaml itself is never committed (data/processed/ is
    # gitignored), so this doesn't reintroduce a portability problem.
    data_yaml = {
        "path": str(OUT_ROOT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }

    data_yaml_path = OUT_ROOT / "data.yaml"

    with open(data_yaml_path, "w") as file:
        yaml.dump(data_yaml, file, sort_keys=False)

    print(f"\ndata.yaml written to: {data_yaml_path}")


def main():
    random.seed(42)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_ROOT}")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    create_folders()

    for split in SPLIT_TARGETS:
        print(f"\nProcessing split: {split}")

        selected_image_ids = set()

        for class_id in sorted(VALID_CLASSES):
            sampled_ids, breakdown = sample_images_for_class(
                split=split,
                class_id=class_id,
                n_samples=SPLIT_TARGETS[split],
            )

            selected_image_ids.update(sampled_ids)

            breakdown_str = ", ".join(f"{c}={n}" for c, n in breakdown.items() if n)
            print(f"Class {class_id}: selected {len(sampled_ids)} images ({breakdown_str})")

        negative_ids = sample_negative_images(
            split=split,
            n_samples=NEGATIVE_TARGETS[split],
            exclude_ids=selected_image_ids,
        )

        print(f"No-damage negatives: selected {len(negative_ids)} images")

        selected_image_ids.update(negative_ids)

        print(f"Unique images selected for {split}: {len(selected_image_ids)}")

        for image_id in selected_image_ids:
            copy_image_and_label(split, image_id)

    count_output()
    write_data_yaml()

    print("\nDone.")
    print(f"Saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
