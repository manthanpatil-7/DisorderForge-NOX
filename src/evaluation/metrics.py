"""Core per-residue metric implementations.

Reference: Benchmark Contract §9.1
Metrics: AUC-PR, AUC-ROC, Fmax, MCC, Balanced Accuracy

Each function accepts (true_labels, predicted_probabilities) and returns a float.
Sanity assertions enforce valid output ranges.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def auc_pr(labels: np.ndarray, probs: np.ndarray) -> float:
    """Per-residue AUC-PR (Average Precision). Benchmark Contract §9.1.

    Args:
        labels: Binary labels (0 or 1). Must contain both classes.
        probs: Predicted probabilities in [0, 1].

    Returns:
        AUC-PR score in [0, 1].

    Raises:
        ValueError: If labels contain only one class.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    _validate_inputs(labels, probs)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "AUC-PR requires both classes. "
            f"Unique labels: {np.unique(labels).tolist()}"
        )

    score = float(np.clip(average_precision_score(labels, probs), 0.0, 1.0))
    return score


def auc_roc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Per-residue AUC-ROC. Benchmark Contract §9.1.

    Args:
        labels: Binary labels (0 or 1). Must contain both classes.
        probs: Predicted probabilities in [0, 1].

    Returns:
        AUC-ROC score in [0, 1].

    Raises:
        ValueError: If labels contain only one class.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    _validate_inputs(labels, probs)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "AUC-ROC requires both classes. "
            f"Unique labels: {np.unique(labels).tolist()}"
        )

    score = float(np.clip(roc_auc_score(labels, probs), 0.0, 1.0))
    return score


def fmax(labels: np.ndarray, probs: np.ndarray) -> float:
    """Maximum F1 score over all thresholds. Benchmark Contract §9.1.

    Sweeps thresholds using the precision-recall curve and computes F1
    at each operating point.

    Args:
        labels: Binary labels (0 or 1).
        probs: Predicted probabilities in [0, 1].

    Returns:
        Fmax (maximum F1) in [0, 1].
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    _validate_inputs(labels, probs)

    if np.sum(labels) == 0:
        return 0.0  # no positives → F1 is 0 at every threshold

    precision, recall, _ = precision_recall_curve(labels, probs)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    score = float(np.clip(np.max(f1_scores), 0.0, 1.0))
    return score


def mcc_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    """Matthews Correlation Coefficient at a fixed threshold. Benchmark Contract §9.1.

    Args:
        labels: Binary labels (0 or 1).
        probs: Predicted probabilities in [0, 1].
        threshold: Decision threshold.

    Returns:
        MCC in [-1, 1].
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    _validate_inputs(labels, probs)

    preds = (probs >= threshold).astype(int)
    score = float(np.clip(matthews_corrcoef(labels, preds), -1.0, 1.0))
    return score


def balanced_accuracy_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> float:
    """Balanced Accuracy at a fixed threshold. Benchmark Contract §9.1.

    Args:
        labels: Binary labels (0 or 1).
        probs: Predicted probabilities in [0, 1].
        threshold: Decision threshold.

    Returns:
        Balanced Accuracy in [0, 1].
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    _validate_inputs(labels, probs)

    preds = (probs >= threshold).astype(int)
    score = float(np.clip(balanced_accuracy_score(labels, preds), 0.0, 1.0))
    return score


def _validate_inputs(labels: np.ndarray, probs: np.ndarray) -> None:
    """Common input validation for all metrics."""
    if len(labels) != len(probs):
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, probs={len(probs)}"
        )
    if len(labels) == 0:
        raise ValueError("Empty inputs")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(
            f"Labels must be binary (0 or 1). Got unique: {np.unique(labels).tolist()}"
        )
    if np.any(np.isnan(probs)):
        raise ValueError("Probabilities contain NaN")
