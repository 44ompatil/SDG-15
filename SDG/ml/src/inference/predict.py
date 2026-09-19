"""
ml/src/inference/predict.py
----------------------------
Standalone inference module.

Accepts a single image (file path or PIL Image) and returns structured
prediction output that is consumed by both the CLI and the FastAPI service.

Output schema:
    {
        "predicted_class": str,
        "confidence": float,          # 0.0 – 1.0
        "is_target_species": bool,
        "model_version": str,
        "all_scores": {class_name: float, ...}
    }

Usage (CLI):
    python -m ml.src.inference.predict --image path/to/image.jpg \\
           --config ml/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.src.config import load_config
from ml.src.training.train import build_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loader (singleton-style – load once, reuse)
# ---------------------------------------------------------------------------

class _ModelStore:
    """Holds a single loaded model so it is not re-loaded per inference call."""

    _instance: "_ModelStore | None" = None

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        target_class: str,
        img_cfg: dict[str, Any],
        model_version: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.target_class = target_class
        self.img_cfg = img_cfg
        self.model_version = model_version
        self.device = device

    @classmethod
    def load(cls, model_path: Path, target_class: str) -> "_ModelStore":
        if cls._instance is not None:
            return cls._instance

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}. Train the model first."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)

        model = build_model(
            architecture=checkpoint["architecture"],
            num_classes=len(checkpoint["class_names"]),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        cls._instance = cls(
            model=model,
            class_names=checkpoint["class_names"],
            target_class=target_class,
            img_cfg=checkpoint["image_config"],
            model_version=checkpoint.get("model_version", "unknown"),
            device=device,
        )
        logger.info(
            "Model loaded: arch=%s  classes=%s  version=%s  device=%s",
            checkpoint["architecture"],
            checkpoint["class_names"],
            cls._instance.model_version,
            device,
        )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached model (useful for tests)."""
        cls._instance = None


# ---------------------------------------------------------------------------
# Public inference function
# ---------------------------------------------------------------------------

def predict(
    image: Path | Image.Image,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run inference on a single image.

    Parameters
    ----------
    image:
        Path to an image file **or** a pre-loaded ``PIL.Image.Image``.
    config:
        Loaded configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Structured prediction result.

    Raises
    ------
    FileNotFoundError
        If *image* is a path that does not exist or the model is not found.
    ValueError
        If the image cannot be opened or is corrupt.
    """
    inference_cfg = config["inference"]
    model_path = Path(inference_cfg["model_path"])
    target_class = config["dataset"]["target_class"]
    threshold = inference_cfg["confidence_threshold"]

    store = _ModelStore.load(model_path, target_class)

    # ---- pre-process ----
    if isinstance(image, Path) or isinstance(image, str):
        image = Path(image)
        if not image.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        try:
            pil_image = Image.open(image).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image}': {exc}") from exc
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Expected Path or PIL.Image, got {type(image)}")

    img_cfg = store.img_cfg
    size = (img_cfg["height"], img_cfg["width"])
    tf = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_cfg["mean"], std=img_cfg["std"]),
        ]
    )

    tensor = tf(pil_image).unsqueeze(0).to(store.device)

    # ---- inference ----
    with torch.no_grad():
        logits = store.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    class_scores = {cls: float(probs[i]) for i, cls in enumerate(store.class_names)}
    predicted_class = max(class_scores, key=class_scores.__getitem__)
    confidence = class_scores[predicted_class]
    is_target = (
        predicted_class == target_class and confidence >= threshold
    )

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "is_target_species": is_target,
        "model_version": store.model_version,
        "all_scores": {k: round(v, 6) for k, v in class_scores.items()},
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a single image"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", default="ml/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    cfg = load_config(args.config)
    result = predict(Path(args.image), cfg)
    print(json.dumps(result, indent=2))
