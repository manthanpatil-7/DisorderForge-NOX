"""Bootstrap confidence interval computation at the protein level.

Reference: Benchmark Contract §10.2

Resampling is at the protein level (not residue level).
Supports paired bootstrap test for competitor comparison.
"""

from typing import Optional

import numpy as np


def bootstrap_ci(
    per_protein_values: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    rng_seed: Optional[int] = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a protein-level metric.

    Args:
        per_protein_values: List of per-protein metric values.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95% CI).
        rng_seed: Optional seed for reproducibility.

    Returns:
        (lower, upper) bounds of the (1-alpha) CI.
    """
    values = np.asarray(per_protein_values, dtype=float)
    n = len(values)

    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(n_resamples)

    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(values[indices])

    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return (lower, upper)


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 10000,
    rng_seed: Optional[int] = None,
) -> float:
    """Paired bootstrap test: is metric A significantly different from B?

    Both value lists must be the same length (one value per protein,
    paired by protein identity).

    Args:
        values_a: Per-protein metric values for method A.
        values_b: Per-protein metric values for method B.
        n_resamples: Number of bootstrap resamples.
        rng_seed: Optional seed for reproducibility.

    Returns:
        Two-sided p-value (proportion of resamples where the sign of
        the difference flips).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Paired test requires equal lengths: {len(a)} vs {len(b)}")

    n = len(a)
    if n == 0:
        return float("nan")

    observed_diff = np.mean(a) - np.mean(b)
    diffs = a - b

    rng = np.random.default_rng(rng_seed)
    count_extreme = 0

    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_diff = np.mean(diffs[indices])
        if abs(boot_diff) >= abs(observed_diff):
            count_extreme += 1

    return count_extreme / n_resamples
