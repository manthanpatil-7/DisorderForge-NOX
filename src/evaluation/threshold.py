"""Frozen-threshold workflow for threshold-dependent diagnostics.

Reference: Benchmark Contract §9.4

Protocol:
1. On internal validation: find threshold that maximizes F1 (argmax F1).
2. Freeze that threshold.
3. On external benchmarks: apply frozen threshold for MCC and BalAcc.
   Do NOT re-optimize on the external set.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Find the threshold that maximizes F1 on the given data.

    Args:
        labels: Binary labels (0 or 1).
        probs: Predicted probabilities in [0, 1].

    Returns:
        Optimal threshold (float).
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    if np.sum(labels) == 0:
        return 0.5  # no positives; default

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    precision = precision[:-1]
    recall = recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def apply_frozen_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a frozen threshold to produce binary predictions.

    Args:
        probs: Predicted probabilities in [0, 1].
        threshold: Frozen threshold from internal validation.

    Returns:
        Binary predictions (0 or 1).
    """
    probs = np.asarray(probs, dtype=float)
    return (probs >= threshold).astype(int)


def compute_metrics_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute MCC and Balanced Accuracy at a frozen threshold.

    Args:
        labels: Binary labels (0 or 1).
        probs: Predicted probabilities in [0, 1].
        threshold: Frozen threshold (NOT re-optimized on this data).

    Returns:
        Dict with 'mcc', 'balanced_accuracy', 'threshold'.
    """
    from src.evaluation.metrics import balanced_accuracy_at_threshold, mcc_at_threshold

    return {
        "mcc": mcc_at_threshold(labels, probs, threshold),
        "balanced_accuracy": balanced_accuracy_at_threshold(labels, probs, threshold),
        "threshold": threshold,
    }
