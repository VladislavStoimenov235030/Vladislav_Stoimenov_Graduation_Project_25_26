"""Adapts a YOLO-format detection dataset (images/ + labels/*.txt, normalized
xywh boxes) to the (image, target) format torchvision detection models expect:
target = {"boxes": FloatTensor[N, 4] in xyxy pixels, "labels": LongTensor[N]}.

Used so the same processed dataset (e.g. data/processed/RDD2022_4classes_medium)
can train both YOLOv8 and torchvision models like Faster R-CNN without
duplicating or reformatting the data on disk.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


class YoloDetectionDataset(Dataset):
    """Reads a YOLO-format split (images/, labels/) into torchvision targets.

    torchvision detection models reserve label 0 for background, so YOLO
    class ids (0-3) are shifted by +1 when building `labels`.
    """

    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.image_paths)

    def _read_boxes(self, label_path: Path, width: int, height: int):
        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id, x_center, y_center, box_w, box_h = (
                int(float(parts[0])),
                *(float(value) for value in parts[1:]),
            )

            x_center *= width
            y_center *= height
            box_w *= width
            box_h *= height

            x_min = x_center - box_w / 2
            y_min = y_center - box_h / 2
            x_max = x_center + box_w / 2
            y_max = y_center + box_h / 2

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id + 1)  # shift: 0 is reserved for background

        return boxes, labels

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes, labels = self._read_boxes(label_path, width, height)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }

        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))
