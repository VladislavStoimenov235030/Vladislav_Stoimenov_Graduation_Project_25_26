from pathlib import Path
import shutil
import random
import yaml


RAW_ROOT = Path("data/raw/RDD2022")
OUT_ROOT = Path("data/processed/RDD2022_4classes_medium")

VALID_CLASSES = {0, 1, 2, 3}

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}

# Per-class positive samples (an image can satisfy more than one class,
# so unique image counts per split are usually slightly lower than the sum).
SPLIT_TARGETS = {
    "train": 250,
    "val": 55,
    "test": 55,
}

# No-damage images added so the model also sees clean road as background.
# This is meant to reduce the false-positive rate the first baseline showed
# (background regions predicted as D00/D40).
NEGATIVE_TARGETS = {
    "train": 50,
    "val": 10,
    "test": 10,
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


def sample_images_for_class(split: str, class_id: int, n_samples: int):
    label_dir = RAW_ROOT / split / "labels"

    candidate_image_ids = []

    for label_path in label_dir.glob("*.txt"):
        classes = get_classes_from_label(label_path)

        if class_id in classes:
            candidate_image_ids.append(label_path.stem)

    if len(candidate_image_ids) < n_samples:
        print(
            f"Warning: split={split}, class={class_id} has only "
            f"{len(candidate_image_ids)} images. Requested {n_samples}."
        )

    n_samples = min(n_samples, len(candidate_image_ids))

    return random.sample(candidate_image_ids, n_samples)


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
            f"Warning: split={split} has only "
            f"{len(candidate_image_ids)} no-damage images. Requested {n_samples}."
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
    print("\nCreated medium dataset:")

    for split in SPLIT_TARGETS:
        image_dir = OUT_ROOT / split / "images"
        label_dir = OUT_ROOT / split / "labels"

        images = list(image_dir.glob("*"))
        labels = list(label_dir.glob("*.txt"))

        negative_labels = sum(
            1 for label_path in label_dir.glob("*.txt") if len(read_label(label_path)) == 0
        )

        print(
            f"{split}: {len(images)} images, {len(labels)} labels "
            f"({negative_labels} with no damage box)"
        )


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
            sampled_ids = sample_images_for_class(
                split=split,
                class_id=class_id,
                n_samples=SPLIT_TARGETS[split],
            )

            selected_image_ids.update(sampled_ids)

            print(f"Class {class_id}: selected {len(sampled_ids)} images")

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
