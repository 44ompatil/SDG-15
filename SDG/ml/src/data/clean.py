"""
ml/src/data/clean.py
--------------------
Applies basic cleaning to raw images before splitting:

* Re-encodes corrupt-but-salvageable images.
* Converts RGBA / palette images to RGB.
* Removes images that are below a minimum resolution.

Usage:
    python -m ml.src.data.clean --config ml/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ml.src.config import load_config

logger = logging.getLogger(__name__)

MIN_DIMENSION = 32  # pixels – images smaller than this are discarded

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_dataset(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Clean raw images in-place (or report only when *dry_run* is True).

    Parameters
    ----------
    config:
        Loaded configuration dictionary.
    dry_run:
        If ``True``, log what would happen without modifying any files.

    Returns
    -------
    dict[str, Any]
        Summary with counts per class.
    """
    raw_dir = Path(config["dataset"]["raw_dir"])
    classes: list[str] = config["dataset"]["classes"]

    summary: dict[str, Any] = {
        "classes": {},
        "total_kept": 0,
        "total_removed": 0,
        "total_converted": 0,
    }

    for cls in classes:
        cls_dir = raw_dir / cls
        if not cls_dir.is_dir():
            logger.warning("Class directory not found, skipping: %s", cls_dir)
            continue

        cls_summary = {"kept": 0, "removed": 0, "converted": 0}

        for img_path in list(cls_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            outcome = _process_image(img_path, dry_run=dry_run)
            if outcome == "removed":
                cls_summary["removed"] += 1
            elif outcome == "converted":
                cls_summary["converted"] += 1
                cls_summary["kept"] += 1
            else:
                cls_summary["kept"] += 1

        summary["classes"][cls] = cls_summary
        summary["total_kept"] += cls_summary["kept"]
        summary["total_removed"] += cls_summary["removed"]
        summary["total_converted"] += cls_summary["converted"]

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_image(path: Path, dry_run: bool) -> str:
    """Inspect and optionally clean a single image file.

    Returns
    -------
    str
        One of ``'kept'``, ``'converted'``, or ``'removed'``.
    """
    try:
        img = Image.open(path)
        img.load()  # force decode
    except (UnidentifiedImageError, Exception) as exc:
        logger.warning("Removing unreadable image %s: %s", path, exc)
        if not dry_run:
            path.unlink(missing_ok=True)
        return "removed"

    w, h = img.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        logger.warning("Removing too-small image %s (%dx%d)", path, w, h)
        if not dry_run:
            path.unlink(missing_ok=True)
        return "removed"

    needs_conversion = img.mode not in ("RGB",)
    if needs_conversion:
        logger.info("Converting %s from mode %s to RGB", path, img.mode)
        if not dry_run:
            rgb_img = img.convert("RGB")
            # Save as JPEG regardless of original format
            new_path = path.with_suffix(".jpg")
            rgb_img.save(new_path, "JPEG", quality=95)
            if new_path != path:
                path.unlink(missing_ok=True)
        return "converted"

    return "kept"


def print_summary(summary: dict[str, Any]) -> None:
    """Print the cleaning summary to stdout."""
    print("\n=== Dataset Cleaning Summary ===")
    print(f"Total kept     : {summary['total_kept']}")
    print(f"Total removed  : {summary['total_removed']}")
    print(f"Total converted: {summary['total_converted']}\n")
    for cls, info in summary["classes"].items():
        print(
            f"  {cls:20s} kept={info['kept']:>5}  "
            f"removed={info['removed']:>3}  converted={info['converted']:>3}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw dataset images"
    )
    parser.add_argument("--config", default="ml/config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report issues without modifying any files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    cfg = load_config(args.config)
    summary = clean_dataset(cfg, dry_run=args.dry_run)
    print_summary(summary)
