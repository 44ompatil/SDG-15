"""
ml/src/training/evaluate.py
----------------------------
Evaluates a trained model against the test set and writes a report.

Usage:
    python -m ml.src.training.evaluate --config ml/config.yaml [--split test]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.src.config import load_config
from ml.src.training.metrics import compute_metrics, format_metrics
from ml.src.training.train import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    config: dict[str, Any],
    split: str = "test",
) -> dict[str, Any]:
    """Run evaluation on *split* and return the metrics dictionary.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.
    split:
        One of ``'train'``, ``'validation'``, or ``'test'``.

    Returns
    -------
    dict[str, Any]
        Metrics dictionary as returned by :func:`compute_metrics` plus
        dataset size counts.
    """
    dataset_cfg = config["dataset"]
    split_dir_map = {
        "train": Path(dataset_cfg["train_dir"]),
        "validation": Path(dataset_cfg["validation_dir"]),
        "test": Path(dataset_cfg["test_dir"]),
    }
    if split not in split_dir_map:
        raise ValueError(f"Invalid split '{split}'. Choose from {list(split_dir_map)}")

    split_dir = split_dir_map[split]
    if not split_dir.is_dir():
        raise RuntimeError(
            f"Split directory '{split_dir}' not found. Run split.py first."
        )

    model_path = Path(config["inference"]["model_path"])
    if not model_path.exists():
        raise RuntimeError(
            f"Model checkpoint not found: {model_path}. Train the model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names: list[str] = checkpoint["class_names"]
    architecture: str = checkpoint["architecture"]
    img_cfg: dict[str, Any] = checkpoint["image_config"]

    logger.info(
        "Loaded checkpoint: arch=%s  classes=%s  model_version=%s",
        architecture,
        class_names,
        checkpoint.get("model_version"),
    )

    model = build_model(architecture, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Data loader
    size = (img_cfg["height"], img_cfg["width"])
    tf = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
        ]
    )
    ds = datasets.ImageFolder(str(split_dir), transform=tf)
    loader = DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # Collect predictions
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["split"] = split
    metrics["num_samples"] = len(y_true)
    metrics["model_version"] = checkpoint.get("model_version", "unknown")

    # Per-split class counts
    class_counts: dict[str, int] = {}
    for cls_idx, cls_name in enumerate(ds.classes):
        count = sum(1 for label in y_true if label == cls_idx)
        class_counts[cls_name] = count
    metrics["class_distribution"] = class_counts

    return metrics


def save_report(metrics: dict[str, Any], output_path: Path) -> None:
    """Save metrics as JSON to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", default="ml/config.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output",
        default="ml/models/evaluation_report.json",
        help="Output JSON path for the metrics report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    cfg = load_config(args.config)
    metrics = evaluate(cfg, split=args.split)
    print(format_metrics(metrics))
    print(f"  Samples evaluated : {metrics['num_samples']}")
    print(f"  Class distribution: {metrics['class_distribution']}")
    save_report(metrics, Path(args.output))
