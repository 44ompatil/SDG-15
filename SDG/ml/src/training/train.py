"""
ml/src/training/train.py
------------------------
Transfer-learning training script.

Supported architectures (configurable in config.yaml):
    mobilenet_v3_small  (default, fastest)
    mobilenet_v3_large
    efficientnet_b0
    resnet18

Usage:
    python -m ml.src.training.train --config ml/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from ml.src.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(architecture: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Build and return a pretrained model with a replaced classification head.

    Parameters
    ----------
    architecture:
        One of the supported model strings.
    num_classes:
        Number of output classes.
    pretrained:
        Whether to load ImageNet weights.

    Returns
    -------
    nn.Module
        Model ready for fine-tuning.
    """
    weights_arg = "DEFAULT" if pretrained else None

    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "resnet18":
        model = models.resnet18(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _get_transforms(config: dict[str, Any]) -> dict[str, transforms.Compose]:
    img_cfg = config["image"]
    size = (img_cfg["height"], img_cfg["width"])
    mean = img_cfg["mean"]
    std = img_cfg["std"]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {"train": train_tf, "validation": val_tf}


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build and return (train_loader, val_loader, class_names).

    Raises
    ------
    RuntimeError
        If the train or validation directories do not exist.
    """
    dataset_cfg = config["dataset"]
    train_dir = Path(dataset_cfg["train_dir"])
    val_dir = Path(dataset_cfg["validation_dir"])

    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise RuntimeError(
                f"Directory '{d}' not found. Run split.py first."
            )

    tfs = _get_transforms(config)
    train_ds = datasets.ImageFolder(str(train_dir), transform=tfs["train"])
    val_ds = datasets.ImageFolder(str(val_dir), transform=tfs["validation"])

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, train_ds.classes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict[str, Any]) -> None:
    """Full training loop: trains, evaluates each epoch, saves best model."""
    training_cfg = config["training"]
    seed = training_cfg["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    train_loader, val_loader, class_names = build_dataloaders(config)
    num_classes = len(class_names)

    logger.info(
        "Classes: %s  |  Train batches: %d  |  Val batches: %d",
        class_names,
        len(train_loader),
        len(val_loader),
    )

    model = build_model(
        architecture=training_cfg["architecture"],
        num_classes=num_classes,
        pretrained=training_cfg["pretrained"],
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )

    best_val_acc: float = 0.0
    best_model_path = Path(training_cfg["best_model_path"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    checkpoint_every = training_cfg["checkpoint_every"]

    for epoch in range(1, training_cfg["epochs"] + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # ---- validate ----
        val_acc, avg_val_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            epoch,
            training_cfg["epochs"],
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            elapsed,
        )

        # ---- save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, class_names, config, best_model_path)
            logger.info("  ✓ New best model saved (val_acc=%.4f)", best_val_acc)

        # ---- periodic checkpoint ----
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            _save_checkpoint(model, class_names, config, ckpt_path)

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate *model* on *loader*. Returns (accuracy, avg_loss)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return correct / total, total_loss / total


def _save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    config: dict[str, Any],
    path: Path,
) -> None:
    """Save model weights and metadata to *path*."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "architecture": config["training"]["architecture"],
            "model_version": config["inference"]["model_version"],
            "image_config": config["image"],
            "dataset_config": {
                "classes": config["dataset"]["classes"],
                "target_class": config["dataset"]["target_class"],
            },
        },
        path,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train invasive species classifier")
    parser.add_argument("--config", default="ml/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    cfg = load_config(args.config)
    train(cfg)
