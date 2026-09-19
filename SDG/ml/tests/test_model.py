"""
ml/tests/test_model.py
----------------------
Unit tests for model building, metrics, and inference.

Note: These tests use tiny in-memory tensors and a freshly initialised
(not pretrained) model so that no network access or GPU is required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.src.training.metrics import compute_metrics, format_metrics
from ml.src.training.train import build_model, set_seed, _save_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSES = ["target_species", "other_animal", "background"]
NUM_CLASSES = len(CLASSES)

BASE_TRAINING_CFG = {
    "architecture": "mobilenet_v3_small",
    "pretrained": False,  # keep tests offline
    "batch_size": 4,
    "learning_rate": 0.001,
    "epochs": 1,
    "num_workers": 0,
    "checkpoint_every": 0,
    "checkpoint_dir": "ml/models/checkpoints",
    "best_model_path": "ml/models/best_model.pt",
    "seed": 42,
}

BASE_IMAGE_CFG = {
    "width": 224,
    "height": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "max_file_size_bytes": 10_485_760,
}

BASE_INFERENCE_CFG = {
    "model_path": "ml/models/best_model.pt",
    "model_version": "v1.0",
    "confidence_threshold": 0.80,
}


def _make_config(model_path: str | Path) -> dict:
    return {
        "dataset": {
            "classes": CLASSES,
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
        "image": BASE_IMAGE_CFG,
        "training": {**BASE_TRAINING_CFG, "best_model_path": str(model_path)},
        "inference": {**BASE_INFERENCE_CFG, "model_path": str(model_path)},
    }


def _save_dummy_checkpoint(model_path: Path) -> None:
    model = build_model("mobilenet_v3_small", NUM_CLASSES, pretrained=False)
    config = _make_config(model_path)
    _save_checkpoint(model, CLASSES, config, model_path)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class TestBuildModel:
    @pytest.mark.parametrize(
        "arch",
        [
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "resnet18",
        ],
    )
    def test_output_shape(self, arch: str) -> None:
        model = build_model(arch, num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, NUM_CLASSES)

    def test_unsupported_arch_raises(self) -> None:
        with pytest.raises(ValueError):
            build_model("unknown_arch", num_classes=3, pretrained=False)

    def test_num_classes_respected(self) -> None:
        for n in (2, 5, 10):
            model = build_model("mobilenet_v3_small", num_classes=n, pretrained=False)
            model.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy)
            assert out.shape[1] == n

    def test_seed_reproducibility(self) -> None:
        set_seed(0)
        m1 = build_model("resnet18", num_classes=3, pretrained=False)
        set_seed(0)
        m2 = build_model("resnet18", num_classes=3, pretrained=False)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y = [0, 1, 2, 0, 1]
        metrics = compute_metrics(y, y, CLASSES)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self) -> None:
        y_true = [0, 0, 0]
        y_pred = [1, 2, 1]
        metrics = compute_metrics(y_true, y_pred, CLASSES)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_confusion_matrix_shape(self) -> None:
        y = [0, 1, 2]
        metrics = compute_metrics(y, y, CLASSES)
        cm = metrics["confusion_matrix"]
        assert len(cm) == NUM_CLASSES
        assert all(len(row) == NUM_CLASSES for row in cm)

    def test_per_class_keys(self) -> None:
        y = [0, 1, 2]
        metrics = compute_metrics(y, y, CLASSES)
        for cls in CLASSES:
            assert cls in metrics["per_class"]

    def test_format_metrics_returns_string(self) -> None:
        y = [0, 1, 2]
        metrics = compute_metrics(y, y, CLASSES)
        text = format_metrics(metrics)
        assert "Accuracy" in text
        assert "Confusion" in text


# ---------------------------------------------------------------------------
# Inference (predict.py)
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_from_pil_image(self, tmp_path: Path) -> None:
        from ml.src.inference.predict import predict, _ModelStore

        _ModelStore.reset()
        model_path = tmp_path / "test_model.pt"
        _save_dummy_checkpoint(model_path)
        config = _make_config(model_path)

        pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = predict(pil_img, config)

        assert "predicted_class" in result
        assert result["predicted_class"] in CLASSES
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["is_target_species"], bool)
        assert "all_scores" in result
        assert set(result["all_scores"].keys()) == set(CLASSES)
        _ModelStore.reset()

    def test_predict_from_file(self, tmp_path: Path) -> None:
        from ml.src.inference.predict import predict, _ModelStore

        _ModelStore.reset()
        model_path = tmp_path / "test_model.pt"
        _save_dummy_checkpoint(model_path)
        config = _make_config(model_path)

        img_path = tmp_path / "test_img.jpg"
        Image.new("RGB", (224, 224)).save(img_path)
        result = predict(img_path, config)
        assert "predicted_class" in result
        _ModelStore.reset()

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        from ml.src.inference.predict import predict, _ModelStore

        _ModelStore.reset()
        model_path = tmp_path / "test_model.pt"
        _save_dummy_checkpoint(model_path)
        config = _make_config(model_path)

        with pytest.raises(FileNotFoundError):
            predict(tmp_path / "does_not_exist.jpg", config)
        _ModelStore.reset()

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        from ml.src.inference.predict import predict, _ModelStore

        _ModelStore.reset()
        config = _make_config(tmp_path / "nonexistent_model.pt")
        pil_img = Image.new("RGB", (224, 224))
        with pytest.raises(FileNotFoundError):
            predict(pil_img, config)
        _ModelStore.reset()

    def test_all_scores_sum_to_one(self, tmp_path: Path) -> None:
        from ml.src.inference.predict import predict, _ModelStore

        _ModelStore.reset()
        model_path = tmp_path / "test_model.pt"
        _save_dummy_checkpoint(model_path)
        config = _make_config(model_path)

        pil_img = Image.new("RGB", (224, 224))
        result = predict(pil_img, config)
        total = sum(result["all_scores"].values())
        assert total == pytest.approx(1.0, abs=1e-4)
        _ModelStore.reset()
