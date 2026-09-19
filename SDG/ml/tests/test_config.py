"""
ml/tests/test_config.py
-----------------------
Unit tests for the config loader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from ml.src.config import load_config


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    base = {
        "dataset": {
            "classes": ["target_species", "other_animal", "background"],
            "target_class": "target_species",
            "raw_dir": "ml/data/raw",
            "train_dir": "ml/data/train",
            "validation_dir": "ml/data/validation",
            "test_dir": "ml/data/test",
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
            "max_file_size_bytes": 10485760,
        },
        "training": {
            "architecture": "mobilenet_v3_small",
            "pretrained": True,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "num_workers": 0,
            "checkpoint_every": 0,
            "checkpoint_dir": "ml/models/checkpoints",
            "best_model_path": "ml/models/best_model.pt",
            "seed": 42,
        },
        "inference": {
            "model_path": "ml/models/best_model.pt",
            "model_version": "v1.0",
            "confidence_threshold": 0.80,
        },
    }
    # Apply overrides (shallow merge per section)
    for section, values in overrides.items():
        if section in base and isinstance(values, dict):
            base[section].update(values)
        else:
            base[section] = values

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(base), encoding="utf-8")
    return config_path


class TestLoadConfig:
    def test_valid_config_loads(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, {})
        cfg = load_config(path)
        assert cfg["dataset"]["classes"] == [
            "target_species",
            "other_animal",
            "background",
        ]

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(": this is not valid yaml :", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(bad_path)

    def test_target_class_not_in_classes_raises(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path, {"dataset": {"target_class": "non_existent_class"}}
        )
        with pytest.raises(ValueError, match="target_class"):
            load_config(path)

    def test_invalid_splits_raise(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path,
            {
                "dataset": {
                    "train_split": 0.80,
                    "validation_split": 0.15,
                    "test_split": 0.15,
                }
            },
        )
        with pytest.raises(ValueError, match="splits must sum"):
            load_config(path)

    def test_unsupported_architecture_raises(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path, {"training": {"architecture": "bert_large"}}
        )
        with pytest.raises(ValueError, match="Unsupported architecture"):
            load_config(path)
