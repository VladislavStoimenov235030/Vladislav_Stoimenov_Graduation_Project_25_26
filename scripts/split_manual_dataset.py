"""Splits the manually collected/annotated Sofia dataset (data/manual/) into
a fine-tuning set and a held-out test set, for the domain-adaptation stage:

1. Fine-tune the RDD2022-large-trained YOLOv8n and Faster R-CNN models on the
   fine-tune split (short, low-LR second training stage).
2. Evaluate both -- with and without fine-tuning -- against the held-out
   split, which neither model ever trains on.

That gives a clean before/after comparison of whether fine-tuning on real
Sofia photos actually improves real-world detection, not just a claim that
it was done.

Target: ~20% held out per class (D40/pothole only has 35 total images in the
manual set, so a flat 20% split without class-awareness could easily starve
it in the test set or leave too few for fine-tuning) -- uses the same
fair-share per-class quota approach as create_rdd2022_4classes_large.py.

Read-only on data/manual: copies into data/manual_split/{finetune,test}/, the
source images/labels are never modified or moved.
"""

from pathlib import Path
import random
import shutil

SRC_ROOT = Path("data/manual")
OUT_ROOT = Path("data/manual_split")

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}

TEST_FRACTION = 0.20
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
    """Same water-filling idea as create_rdd2022_4classes_large.py, but here
    each class has its own target (not a shared one) since the manual
    dataset's classes are very unevenly sized (150 vs 35 images).
    """
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
    random.seed(42)

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Manual dataset not found: {SRC_ROOT}")

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    label_dir = SRC_ROOT / "labels"
    candidates_by_class = {cid: [] for cid in CLASS_NAMES}

    for label_path in label_dir.glob("*.txt"):
        for cid in set(get_classes_from_label(label_path)):
            if cid in candidates_by_class:
                candidates_by_class[cid].append(label_path.stem)

    print("Class availability in data/manual:")
    for cid, name in CLASS_NAMES.items():
        print(f"   {cid} {name:25s} {len(candidates_by_class[cid])} images")

    n_target_per_class = {
        cid: max(round(len(ids) * TEST_FRACTION), 1) for cid, ids in candidates_by_class.items()
    }
    print(f"\nPer-class test targets (~{TEST_FRACTION:.0%}):")
    for cid, name in CLASS_NAMES.items():
        print(f"   {cid} {name:25s} {n_target_per_class[cid]}")

    test_stems = fair_share_sample(candidates_by_class, n_target_per_class)

    all_stems = {p.stem for p in label_dir.glob("*.txt")}
    finetune_stems = all_stems - test_stems

    print(f"\nTest (held-out):  {len(test_stems)} images")
    print(f"Fine-tune:        {len(finetune_stems)} images")

    for stem in sorted(test_stems):
        copy_pair(stem, "test")
    for stem in sorted(finetune_stems):
        copy_pair(stem, "finetune")

    # Per-split class distribution, so the balance can be checked directly.
    for split, stems in (("test", test_stems), ("finetune", finetune_stems)):
        counts = {cid: 0 for cid in CLASS_NAMES}
        for stem in stems:
            for cid in set(get_classes_from_label(label_dir / f"{stem}.txt")):
                if cid in counts:
                    counts[cid] += 1
        print(f"\n{split} class distribution:")
        for cid, name in CLASS_NAMES.items():
            print(f"   {cid} {name:25s} {counts[cid]}")

    print(f"\nDone. Saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
