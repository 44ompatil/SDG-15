"""
ml/src/config.py
----------------
Single entry-point for loading and validating the YAML configuration.
All other modules should import from here instead of reading YAML directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "ml/config.yaml") -> dict[str, Any]:
    """Load and return the YAML configuration as a plain dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file. Defaults to ``ml/config.yaml``
        relative to the current working directory.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the YAML file cannot be parsed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        try:
            config: dict[str, Any] = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse configuration file: {exc}") from exc

    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Perform basic sanity checks on the loaded configuration.

    Raises
    ------
    ValueError
        If a required key is missing or a value is clearly invalid.
    """
    required_sections = ("dataset", "image", "training", "inference")
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    dataset = config["dataset"]
    if not dataset.get("classes"):
        raise ValueError("dataset.classes must be a non-empty list")

    target = dataset.get("target_class")
    if target not in dataset["classes"]:
        raise ValueError(
            f"dataset.target_class '{target}' is not in dataset.classes {dataset['classes']}"
        )

    splits = (
        dataset.get("train_split", 0),
        dataset.get("validation_split", 0),
        dataset.get("test_split", 0),
    )
    total = sum(splits)
    if not (0.999 < total < 1.001):
        raise ValueError(
            f"Train/validation/test splits must sum to 1.0 (got {total:.4f})"
        )

    training = config["training"]
    supported_architectures = (
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "resnet18",
    )
    arch = training.get("architecture")
    if arch not in supported_architectures:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Choose from: {supported_architectures}"
        )
