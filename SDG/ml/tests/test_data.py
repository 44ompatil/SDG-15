"""
ml/tests/test_data.py
---------------------
Unit tests for the dataset pipeline (collect, clean, split).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from ml.src.data.collect import validate_dataset, _check_image, _file_hash
from ml.src.data.clean import clean_dataset
from ml.src.data.split import split_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSES = ["target_species", "other_animal"]


def _make_rgb_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    """Save a small valid RGB JPEG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 150, 200)).save(path, "JPEG")


def _make_rgba_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size).save(path, "PNG")


def _corrupt_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01corrupt data")


@pytest.fixture()
def base_config(tmp_path: Path) -> dict:
    return {
        "dataset": {
            "classes": CLASSES,
            "target_class": "target_species",
            "raw_dir": str(tmp_path / "raw"),
            "train_dir": str(tmp_path / "train"),
            "validation_dir": str(tmp_path / "validation"),
            "test_dir": str(tmp_path / "test"),
            "train_split": 0.70,
            "validation_split": 0.15,
            "test_split": 0.15,
            "seed": 42,
        },
        "image": {
            "width": 224,
            "height": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "max_file_size_bytes": 10_485_760,
        },
        "training": {
            "architecture": "mobilenet_v3_small",
            "batch_size": 4,
            "num_workers": 0,
        },
    }


# ---------------------------------------------------------------------------
# collect.py tests
# ---------------------------------------------------------------------------

class TestValidateDataset:
    def test_missing_raw_dir(self, base_config: dict, tmp_path: Path) -> None:
        report = validate_dataset(base_config)
        assert any("does not exist" in e for e in report["errors"])

    def test_valid_images_counted(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        for cls in CLASSES:
            for i in range(3):
                _make_rgb_image(raw / cls / f"img_{i}.jpg")
        report = validate_dataset(base_config)
        assert report["total_valid"] == 6
        assert report["total_corrupt"] == 0

    def test_corrupt_images_flagged(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        _make_rgb_image(raw / "target_species" / "good.jpg")
        _corrupt_image(raw / "target_species" / "bad.jpg")
        _make_rgb_image(raw / "other_animal" / "good.jpg")
        report = validate_dataset(base_config)
        assert report["total_corrupt"] == 1
        assert report["total_valid"] == 2

    def test_duplicate_detection(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        src = raw / "target_species" / "img1.jpg"
        _make_rgb_image(src)
        dup = raw / "other_animal" / "img1_dup.jpg"
        dup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dup)
        report = validate_dataset(base_config)
        assert len(report["duplicate_hashes"]) == 1

    def test_check_image_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "valid.jpg"
        _make_rgb_image(path)
        ok, msg = _check_image(path, 10_485_760)
        assert ok
        assert msg == ""

    def test_check_image_corrupt(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        _corrupt_image(path)
        ok, msg = _check_image(path, 10_485_760)
        assert not ok

    def test_check_image_too_large(self, tmp_path: Path) -> None:
        path = tmp_path / "large.jpg"
        _make_rgb_image(path)
        ok, msg = _check_image(path, max_bytes=1)
        assert not ok
        assert "limit" in msg


# ---------------------------------------------------------------------------
# clean.py tests
# ---------------------------------------------------------------------------

class TestCleanDataset:
    def test_rgba_converted_to_rgb(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        rgba_path = raw / "target_species" / "rgba_img.png"
        _make_rgba_image(rgba_path)
        summary = clean_dataset(base_config, dry_run=False)
        assert summary["total_converted"] == 1

    def test_corrupt_removed(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        _corrupt_image(raw / "target_species" / "bad.jpg")
        summary = clean_dataset(base_config, dry_run=False)
        assert summary["total_removed"] == 1

    def test_dry_run_does_not_modify(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        corrupt_path = raw / "target_species" / "bad.jpg"
        _corrupt_image(corrupt_path)
        clean_dataset(base_config, dry_run=True)
        assert corrupt_path.exists(), "dry_run must not delete files"


# ---------------------------------------------------------------------------
# split.py tests
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def _populate_raw(self, raw: Path, n: int = 20) -> None:
        for cls in CLASSES:
            for i in range(n):
                _make_rgb_image(raw / cls / f"img_{i:03d}.jpg")

    def test_split_creates_correct_dirs(
        self, base_config: dict, tmp_path: Path
    ) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        self._populate_raw(raw, n=20)
        summary = split_dataset(base_config, dry_run=False)
        train_dir = Path(base_config["dataset"]["train_dir"])
        assert (train_dir / "target_species").is_dir()

    def test_no_data_leakage(self, base_config: dict, tmp_path: Path) -> None:
        raw = Path(base_config["dataset"]["raw_dir"])
        self._populate_raw(raw, n=30)
        split_dataset(base_config, dry_run=False)

        train_files = set(
            p.name
            for cls in CLASSES
            for p in (Path(base_config["dataset"]["train_dir"]) / cls).glob("*")
        )
        test_files = set(
            p.name
            for cls in CLASSES
            for p in (Path(base_config["dataset"]["test_dir"]) / cls).glob("*")
        )
        assert train_files.isdisjoint(test_files), "Data leakage detected!"

    def test_total_counts_match(self, base_config: dict, tmp_path: Path) -> None:
        n_per_class = 20
        raw = Path(base_config["dataset"]["raw_dir"])
        self._populate_raw(raw, n=n_per_class)
        summary = split_dataset(base_config, dry_run=False)
        for cls, counts in summary["classes"].items():
            total = sum(counts.values())
            assert total == n_per_class

    def test_reproducibility(self, base_config: dict, tmp_path: Path) -> None:
        """Same seed → same split twice (dry run)."""
        raw = Path(base_config["dataset"]["raw_dir"])
        self._populate_raw(raw, n=20)
        s1 = split_dataset(base_config, dry_run=True)
        s2 = split_dataset(base_config, dry_run=True)
        assert s1 == s2
