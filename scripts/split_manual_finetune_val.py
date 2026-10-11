"""Carves a small validation slice out of data/manual_split/finetune/, so the
fine-tuning stage (notebooks 11/12) can monitor for overfitting without ever
touching data/manual_split/test/ (the fully held-out before/after comparison
set built by split_manual_dataset.py).

~18% held out per class as fine-tune-val, same fair-share per-class approach
as the other manual-dataset splitting script. Read-only on
data/manual_split/finetune -- copies into data/manual_finetune_split/{train,val}/.
"""

from pathlib import Path
import random
import shutil

SRC_ROOT = Path("data/manual_split/finetune")
OUT_ROOT = Path("data/manual_finetune_split")

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}

VAL_FRACTION = 0.18
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_classes_from_label(label_path: Path) -> list[int]:
    classes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        classes.append(int(float(parts[0])))
    return classes


def fair_share_sample(candidates_by_class: dict, n_target_per_class: dict) -> set:
    selected = set()
    for class_id, candidates in candidates_by_class.items():
        target = n_target_per_class[class_id]
        already = [c for c in candidates if c in selected]
        remaining_target = max(target - len(already), 0)
        remaining_pool = [c for c in candidates if c not in selected]
        take = min(remaining_target, len(remaining_pool))
        if take > 0:
            selected.update(random.sample(remaining_pool, take))
    return selected


def copy_pair(stem: str, dest_split: str):
    image_path = None
    for ext in IMAGE_EXTENSIONS:
        candidate = SRC_ROOT / "images" / f"{stem}{ext}"
        if candidate.exists():
            image_path = candidate
            break
    label_path = SRC_ROOT / "labels" / f"{stem}.txt"

    if image_path is None or not label_path.exists():
        print(f"Warning: missing image or label for {stem}, skipping")
        return

    out_image_dir = OUT_ROOT / dest_split / "images"
    out_label_dir = OUT_ROOT / dest_split / "labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, out_image_dir / image_path.name)
    shutil.copy2(label_path, out_label_dir / label_path.name)


def main():
    random.seed(43)  # different seed from split_manual_dataset.py -- independent split

    if not SRC_ROOT.exists():
        raise FileNotFoundError(
            f"Not found: {SRC_ROOT}. Run scripts/split_manual_dataset.py first."
        )

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    label_dir = SRC_ROOT / "labels"
    candidates_by_class = {cid: [] for cid in CLASS_NAMES}

    for label_path in label_dir.glob("*.txt"):
        for cid in set(get_classes_from_label(label_path)):
            if cid in candidates_by_class:
                candidates_by_class[cid].append(label_path.stem)

    n_target_per_class = {
        cid: max(round(len(ids) * VAL_FRACTION), 1) for cid, ids in candidates_by_class.items()
    }

    val_stems = fair_share_sample(candidates_by_class, n_target_per_class)
    all_stems = {p.stem for p in label_dir.glob("*.txt")}
    train_stems = all_stems - val_stems

    print(f"fine-tune train: {len(train_stems)} images")
    print(f"fine-tune val:   {len(val_stems)} images")

    for stem in sorted(train_stems):
        copy_pair(stem, "train")
    for stem in sorted(val_stems):
        copy_pair(stem, "val")

    print(f"\nDone. Saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
