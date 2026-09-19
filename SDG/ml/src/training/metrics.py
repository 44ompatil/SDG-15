"""
ml/src/training/metrics.py
--------------------------
Computes and formats evaluation metrics for multi-class classification.

All metric computation is kept here so it can be imported by both
evaluate.py and tests without duplicating logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute a full set of classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_pred:
        Predicted class indices.
    class_names:
        Ordered list of class names (index → name).

    Returns
    -------
    dict[str, Any]
        Dictionary with accuracy, precision, recall, F1, confusion matrix,
        and per-class metrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    avg_kwargs = dict(average="macro", zero_division=0)
    precision = float(precision_score(y_true, y_pred, **avg_kwargs))
    recall = float(recall_score(y_true, y_pred, **avg_kwargs))
    f1 = float(f1_score(y_true, y_pred, **avg_kwargs))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    per_class_report: dict[str, dict[str, float]] = {}
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    for cls in class_names:
        if cls in report:
            per_class_report[cls] = {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": int(report[cls]["support"]),
            }

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_class": per_class_report,
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Return a human-readable metrics report string."""
    lines = [
        "\n=== Evaluation Metrics ===",
        f"  Accuracy        : {metrics['accuracy']:.4f}",
        f"  Precision (macro): {metrics['precision_macro']:.4f}",
        f"  Recall (macro)  : {metrics['recall_macro']:.4f}",
        f"  F1 (macro)      : {metrics['f1_macro']:.4f}",
        "",
        "  Per-class metrics:",
    ]
    for cls, info in metrics["per_class"].items():
        lines.append(
            f"    {cls:<22s}  P={info['precision']:.3f}  "
            f"R={info['recall']:.3f}  F1={info['f1']:.3f}  "
            f"n={info['support']}"
        )

    lines.append("\n  Confusion matrix (rows=true, cols=pred):")
    class_names = metrics["class_names"]
    header = "  " + " " * 22 + "  ".join(f"{c[:8]:>8}" for c in class_names)
    lines.append(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>8}" for v in row)
        lines.append(f"  {class_names[i]:<22s}  {row_str}")
    lines.append("")
    return "\n".join(lines)
