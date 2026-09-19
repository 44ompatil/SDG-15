# Machine Learning – Invasive Animal Detection

## Overview

Transfer-learning image classifier built on PyTorch.
Supports **MobileNetV3-Small** (default), MobileNetV3-Large, EfficientNet-B0, and ResNet-18.

## Quick-start

```bash
# 1. Install dependencies
pip install -r ml/requirements.txt

# 2. Populate ml/data/raw/<class_name>/  with your images, then:

# 3. Validate the raw dataset
python -m ml.src.data.collect --config ml/config.yaml

# 4. Clean (remove corrupt/tiny images, convert RGBA → RGB)
python -m ml.src.data.clean --config ml/config.yaml

# 5. Split into train / validation / test
python -m ml.src.data.split --config ml/config.yaml

# 6. Train
python -m ml.src.training.train --config ml/config.yaml

# 7. Evaluate on test set
python -m ml.src.training.evaluate --config ml/config.yaml --split test

# 8. Run inference on a single image
python -m ml.src.inference.predict --image path/to/image.jpg --config ml/config.yaml
```

## Configuration

All behaviour is controlled by [`ml/config.yaml`](../ml/config.yaml).
Key parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.classes` | `[target_species, other_animal, background]` | Class names (match raw/ subdirectories) |
| `dataset.target_class` | `target_species` | The invasive species class |
| `training.architecture` | `mobilenet_v3_small` | Model architecture |
| `training.epochs` | `20` | Training epochs |
| `training.learning_rate` | `0.001` | Initial LR (ReduceLROnPlateau scheduler) |
| `inference.confidence_threshold` | `0.80` | Min confidence to trigger an alert |

## Data Layout

```
ml/data/
├── raw/
│   ├── target_species/   ← your invasive animal images
│   ├── other_animal/     ← non-target animal images
│   └── background/       ← empty scene / background images
├── train/
├── validation/
└── test/
```

## Running Tests

```bash
# From the project root:
pytest ml/tests/ -v
```

## Implementation Status

| Feature | Status |
|---------|--------|
| Config loader + validation | ✅ |
| Dataset validation (collect.py) | ✅ |
| Image cleaning (clean.py) | ✅ |
| Train/val/test split (split.py) | ✅ |
| Transfer-learning training (train.py) | ✅ |
| Metrics module (metrics.py) | ✅ |
| Evaluation script (evaluate.py) | ✅ |
| Inference / predict.py | ✅ |
| Unit tests | ✅ |
| Real trained model | ⏳ Requires dataset |
