"""Ensemble prediction utilities.

Reference: Training Contract §16.1

Simple averaging and learned-weight fusion for combining
expert predictions into a single disorder probability.
"""

import numpy as np


def average_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    """Average per-residue predictions from multiple experts.

    Args:
        predictions: List of 1D arrays, each shape [L] with probabilities.
            All must have the same length.

    Returns:
        Averaged probability array, shape [L].

    Raises:
        ValueError: If predictions have mismatched lengths or list is empty.
    """
    if not predictions:
        raise ValueError("Empty prediction list")

    ref_len = len(predictions[0])
    for i, pred in enumerate(predictions):
        if len(pred) != ref_len:
            raise ValueError(
                f"Shape mismatch: prediction 0 has length {ref_len}, "
                f"prediction {i} has length {len(pred)}"
            )

    return np.mean(predictions, axis=0)
