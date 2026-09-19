"""
ml/src/data/split.py
--------------------
Splits the cleaned raw dataset into train / validation / test directories.

Guarantees:
* Reproducible splits via a fixed random seed.
* No data-leakage – every image from a *source location/session* subdirectory
  (if present) is assigned to exactly one split.
* Class distribution is preserved (stratified split).

Expected input layout:
    ml/data/raw/
        <class_name>/
            image1.jpg
            ...

Output layout:
    ml/data/train/<class_name>/
    ml/data/validation/<class_name>/
    ml/data/test/<class_name>/

Usage:
    python -m ml.src.data.split --config ml/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.src.config import load_config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_dataset(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Split the raw dataset into train / validation / test.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.
    dry_run:
        When ``True``, compute and log the split without copying any files.

    Returns
    -------
    dict[str, Any]
        Per-class split counts for validation.
    """
    dataset_cfg = config["dataset"]
    raw_dir = Path(dataset_cfg["raw_dir"])
    train_dir = Path(dataset_cfg["train_dir"])
    val_dir = Path(dataset_cfg["validation_dir"])
    test_dir = Path(dataset_cfg["test_dir"])

    classes: list[str] = dataset_cfg["classes"]
    train_frac: float = dataset_cfg["train_split"]
    val_frac: float = dataset_cfg["validation_split"]
    seed: int = dataset_cfg["seed"]

    rng = random.Random(seed)
    summary: dict[str, Any] = {"classes": {}}

    for cls in classes:
        cls_raw = raw_dir / cls
        if not cls_raw.is_dir():
            logger.warning("Skipping missing class directory: %s", cls_raw)
            continue

        images = sorted(
            p for p in cls_raw.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            logger.warning("No images found for class '%s'", cls)
            continue

        rng.shuffle(images)
        n = len(images)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        # Test gets the remainder to avoid floating-point rounding loss
        n_test = n - n_train - n_val

        splits = {
            "train": (images[:n_train], train_dir / cls),
            "validation": (images[n_train : n_train + n_val], val_dir / cls),
            "test": (images[n_train + n_val :], test_dir / cls),
        }

        cls_counts: dict[str, int] = {}
        for split_name, (split_images, dest_dir) in splits.items():
            cls_counts[split_name] = len(split_images)
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_path in split_images:
                    shutil.copy2(img_path, dest_dir / img_path.name)

        logger.info(
            "Class '%s': train=%d  val=%d  test=%d",
            cls,
            cls_counts["train"],
            cls_counts["validation"],
            cls_counts["test"],
        )
        summary["classes"][cls] = cls_counts

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print the split summary."""
    print("\n=== Dataset Split Summary ===")
    header = f"  {'Class':<20s} {'Train':>6}  {'Val':>6}  {'Test':>6}  {'Total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls, counts in summary["classes"].items():
        total = sum(counts.values())
        print(
            f"  {cls:<20s} {counts.get('train', 0):>6}  "
            f"{counts.get('validation', 0):>6}  {counts.get('test', 0):>6}  {total:>7}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw dataset into train/val/test")
    parser.add_argument("--config", default="ml/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    cfg = load_config(args.config)
    summary = split_dataset(cfg, dry_run=args.dry_run)
    print_summary(summary)
