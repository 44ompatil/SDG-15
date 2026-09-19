"""
ml/src/data/collect.py
----------------------
Validates the raw dataset directory before any processing.

Expected layout:
    ml/data/raw/
        <class_name_1>/
            image1.jpg
            image2.jpg
        <class_name_2>/
            ...

Usage:
    python -m ml.src.data.collect --config ml/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

from ml.src.config import load_config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def validate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the raw dataset directory structure and individual files.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.

    Returns
    -------
    dict[str, Any]
        A summary report with counts, errors, and warnings.
    """
    raw_dir = Path(config["dataset"]["raw_dir"])
    classes: list[str] = config["dataset"]["classes"]

    report: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "classes": {},
        "errors": [],
        "warnings": [],
        "total_valid": 0,
        "total_corrupt": 0,
        "duplicate_hashes": [],
    }

    if not raw_dir.exists():
        msg = f"Raw directory does not exist: {raw_dir}"
        logger.error(msg)
        report["errors"].append(msg)
        return report

    seen_hashes: dict[str, list[str]] = {}

    for cls in classes:
        cls_dir = raw_dir / cls
        class_report: dict[str, Any] = {
            "path": str(cls_dir),
            "exists": cls_dir.is_dir(),
            "valid": 0,
            "corrupt": 0,
            "skipped": 0,
        }

        if not cls_dir.is_dir():
            msg = f"Class directory missing: {cls_dir}"
            logger.warning(msg)
            report["warnings"].append(msg)
            report["classes"][cls] = class_report
            continue

        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                class_report["skipped"] += 1
                continue

            ok, err = _check_image(img_path, config["image"]["max_file_size_bytes"])
            if not ok:
                class_report["corrupt"] += 1
                report["total_corrupt"] += 1
                report["errors"].append(f"{img_path}: {err}")
                continue

            file_hash = _file_hash(img_path)
            seen_hashes.setdefault(file_hash, []).append(str(img_path))
            class_report["valid"] += 1
            report["total_valid"] += 1

        report["classes"][cls] = class_report

    # Identify duplicates
    for h, paths in seen_hashes.items():
        if len(paths) > 1:
            report["duplicate_hashes"].append(paths)
            report["warnings"].append(f"Duplicate images found: {paths}")

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_image(path: Path, max_bytes: int) -> tuple[bool, str]:
    """Return (True, '') if image is valid, else (False, reason)."""
    try:
        from PIL import Image, UnidentifiedImageError  # lazy import

        if path.stat().st_size > max_bytes:
            return False, f"File size {path.stat().st_size} exceeds limit {max_bytes}"

        with Image.open(path) as img:
            img.verify()  # checks header integrity
        return True, ""

    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _file_hash(path: Path) -> str:
    """Return a fast MD5 hex digest of *path* for duplicate detection."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print the validation report to stdout."""
    print("\n=== Dataset Validation Report ===")
    print(f"Raw directory : {report['raw_dir']}")
    print(f"Total valid   : {report['total_valid']}")
    print(f"Total corrupt : {report['total_corrupt']}")
    print(f"Duplicates    : {len(report['duplicate_hashes'])} group(s)\n")

    for cls, info in report["classes"].items():
        exists = "✓" if info["exists"] else "✗"
        print(
            f"  [{exists}] {cls:20s} valid={info['valid']:>5}  "
            f"corrupt={info['corrupt']:>3}  skipped={info['skipped']:>3}"
        )

    if report["errors"]:
        print(f"\n⚠  Errors ({len(report['errors'])}):")
        for e in report["errors"][:20]:
            print(f"   {e}")

    if report["warnings"]:
        print(f"\n⚠  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"][:20]:
            print(f"   {w}")
    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the raw dataset directory")
    parser.add_argument(
        "--config", default="ml/config.yaml", help="Path to config YAML"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    cfg = load_config(args.config)
    rpt = validate_dataset(cfg)
    print_report(rpt)
