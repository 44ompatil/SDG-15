#!/usr/bin/env python
"""Run the wildlife checkpoint on the computer receiving ESP32 captures."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.src.training.train import build_model


class WildlifeClassifier:
    """Load the trained checkpoint once and classify camera JPEGs on the host."""

    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.class_names: list[str] = checkpoint["class_names"]
        self.model_version: str = checkpoint.get("model_version", "unknown")
        self.image_config: dict[str, Any] = checkpoint["image_config"]
        dataset_config = checkpoint.get("dataset_config", {})
        self.target_class = dataset_config.get("target_class", "wild_boar")
        self.alert_classes = set(dataset_config.get("alert_classes", []))
        self.confidence_threshold = 0.70

        self.model = build_model(
            checkpoint["architecture"],
            num_classes=len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        image_size = (self.image_config["height"], self.image_config["width"])
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.image_config["mean"],
                    std=self.image_config["std"],
                ),
            ]
        )

    def classify(self, jpeg: bytes) -> dict[str, Any]:
        """Classify one JPEG and return probabilities and alert flags."""
        try:
            with Image.open(io.BytesIO(jpeg)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("uploaded body is not a readable JPEG image") from exc

        model_input = self.transform(image).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(model_input)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        inference_ms = round((time.perf_counter() - start) * 1000)

        if not np.isfinite(probabilities).all():
            raise RuntimeError("classifier returned a non-finite probability")

        predicted_index = int(np.argmax(probabilities))
        species = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        confident = confidence >= self.confidence_threshold
        scores = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.class_names)
        }
        return {
            "species": species,
            "confidence": confidence,
            "model_version": self.model_version,
            "is_alert": confident and species in self.alert_classes,
            "is_target_species": confident and species == self.target_class,
            "is_human": confident and species == "human",
            "inference_ms": inference_ms,
            "scores": scores,
        }
