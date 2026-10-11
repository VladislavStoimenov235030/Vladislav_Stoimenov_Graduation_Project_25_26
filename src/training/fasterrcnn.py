"""Reusable, resource-light Faster R-CNN training/evaluation helpers.

Built so the same recipe can run on the small dataset first and the medium
dataset later with identical code (only DATASET_DIR/hyperparameters change),
the same way the YOLOv8n notebooks (03, 05) reuse Ultralytics' `model.train`.

Kept deliberately light for local, single-GPU training (8GB VRAM class):
- torchvision's default internal resize (~800/1333px) is replaced with a
  fixed size matching YOLO's `imgsz` (default 640), which is the single
  biggest memory/time lever for this model.
- Automatic mixed precision (AMP) is used during training.
- Optional gradient accumulation lets you keep a larger *effective* batch
  size while the real per-step batch (and its memory footprint) stays small.
- Early stopping on validation loss mirrors YOLO's `patience` behaviour.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from src.data.yolo_torchvision_dataset import YoloDetectionDataset, collate_fn

# torchvision detection models reserve label 0 for background; class ids here
# match src.data.yolo_torchvision_dataset's +1 shift of the YOLO class ids.
CLASS_NAMES = {
    1: "D00_Longitudinal_Crack",
    2: "D10_Transverse_Crack",
    3: "D20_Alligator_Crack",
    4: "D40_Pothole",
}
NUM_CLASSES = len(CLASS_NAMES) + 1  # + background


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(imgsz: int = 640, num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN (ResNet50-FPN v2), head replaced for our
    classes, internal resize capped to `imgsz` (== YOLO's imgsz) instead of
    torchvision's much larger default -- the main resource-usage lever.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=imgsz,
        max_size=imgsz,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_dataloaders(
    dataset_dir: Path,
    batch_size: int = 2,
    workers: int = 0,
):
    dataset_dir = Path(dataset_dir)
    splits = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        ds = YoloDetectionDataset(dataset_dir / split / "images", dataset_dir / split / "labels")
        splits[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
    return splits["train"], splits["val"], splits["test"]


@dataclass
class TrainConfig:
    epochs: int = 30
    patience: int = 10  # early stop if val_loss doesn't improve for N epochs
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    step_size: int = 10
    gamma: float = 0.1
    accum_steps: int = 4  # gradient accumulation: effective batch = batch_size * accum_steps
    amp: bool = True  # mixed precision -- lowers memory + time on CUDA
    grad_clip: float = 5.0


def _run_epoch(model, loader, optimizer, device, scaler, train: bool, cfg: TrainConfig):
    model.train()  # Faster R-CNN only returns the loss dict in train() mode
    total_loss, n_batches = 0.0, 0

    if train:
        optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if train:
                scaled = loss / cfg.accum_steps
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        optimizer.step()
                    optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    run_dir: Path,
    cfg: TrainConfig = TrainConfig(),
    progress: bool = True,
):
    """Trains with early stopping on val_loss; saves last.pt/best.pt/results.csv
    into run_dir, mirroring the YOLO run-folder convention.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.gamma
    )
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")

    iterator = range(1, cfg.epochs + 1)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="epochs", unit="epoch")

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in iterator:
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, optimizer, device, scaler, train=True, cfg=cfg)
        val_loss = _run_epoch(model, val_loader, optimizer, device, scaler, train=False, cfg=cfg)
        lr_scheduler.step()
        elapsed = time.time() - t0

        improved = val_loss < best_val_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time_s": elapsed,
            }
        )

        marker = "  <-- best" if improved else ""
        msg = f"epoch {epoch:>3}/{cfg.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.0f}s){marker}"
        if progress:
            iterator.set_postfix(train=f"{train_loss:.3f}", val=f"{val_loss:.3f}")
        else:
            print(msg)

        torch.save(model.state_dict(), run_dir / "last.pt")
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.patience:
            print(
                f"Early stopping: no val_loss improvement for {cfg.patience} epochs (epoch {epoch})."
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "results.csv", index=False)
    return history_df


def _match_counts(p_boxes, p_labels, t_boxes, t_labels, iou_threshold: float):
    """Greedy IoU matching (predictions already sorted/filtered by the caller)
    -> (tp, fp, fn) for one image at one confidence threshold.
    """
    matched_gt = set()
    tp = fp = 0

    if len(p_boxes) and len(t_boxes):
        ious = box_iou(p_boxes, t_boxes)

    for i in range(len(p_boxes)):
        match = -1
        if len(t_boxes):
            best_iou, best_j = 0.0, -1
            for j in range(len(t_boxes)):
                if j in matched_gt or t_labels[j] != p_labels[i]:
                    continue
                if ious[i, j] > best_iou:
                    best_iou, best_j = ious[i, j].item(), j
            if best_iou >= iou_threshold:
                match = best_j
        if match >= 0:
            tp += 1
            matched_gt.add(match)
        else:
            fp += 1

    fn = len(t_boxes) - len(matched_gt)
    return tp, fp, fn


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    iou_threshold: float = 0.5,
    thresholds=tuple(round(t, 2) for t in [i / 100 for i in range(5, 96, 5)]),
):
    """Computes mAP50 / mAP50-95 (torchmetrics) plus precision/recall/F1 at the
    *best-F1 confidence threshold* -- the same idea as the single-number P/R
    YOLO's val() reports (its curve's best operating point), rather than an
    arbitrary fixed threshold that can look artificially perfect-but-empty on
    an undertrained model (100% precision at ~0% recall).
    """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, backend="faster_coco_eval")

    inference_times = []
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]

        t0 = time.time()
        preds = model(images)
        inference_times.append((time.time() - t0) / len(images))

        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds_cpu, targets)
        all_preds.extend(preds_cpu)
        all_targets.extend(targets)

    results = metric.compute()

    best = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "threshold": None}
    for thresh in thresholds:
        tp = fp = fn = 0
        for pred, target in zip(all_preds, all_targets):
            keep = pred["scores"] >= thresh
            tp_i, fp_i, fn_i = _match_counts(
                pred["boxes"][keep],
                pred["labels"][keep],
                target["boxes"],
                target["labels"],
                iou_threshold,
            )
            tp += tp_i
            fp += fp_i
            fn += fn_i

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        if f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1, "threshold": thresh}

    return {
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "best_threshold": best["threshold"],
        "mAP50": results["map_50"].item(),
        "mAP50-95": results["map"].item(),
        "avg_inference_ms": sum(inference_times) / len(inference_times) * 1000,
        "raw_torchmetrics": results,
    }
