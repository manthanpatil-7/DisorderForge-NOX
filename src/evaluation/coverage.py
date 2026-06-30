"""Prediction coverage computation.

Reference: Benchmark Contract §11

Coverage = fraction of benchmark proteins for which the predictor
produced predictions. Uncovered proteins are logged with reasons.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageResult:
    """Result of coverage computation."""

    total_proteins: int
    covered_proteins: int
    coverage_pct: float
    failure_log: list[dict] = field(default_factory=list)
    # Each entry: {"accession": str, "reason": str}


def compute_coverage(
    benchmark: dict,
    predictions: dict[str, np.ndarray],
) -> CoverageResult:
    """Compute prediction coverage over a benchmark set.

    Args:
        benchmark: {accession: BenchmarkProtein} from benchmark_io.
        predictions: {accession: probability_array}.

    Returns:
        CoverageResult with total, covered, coverage %, and failure log.
    """
    total = len(benchmark)
    failure_log = []

    for acc in benchmark:
        if acc not in predictions:
            failure_log.append({"accession": acc, "reason": "no prediction"})
        elif predictions[acc] is None:
            failure_log.append({"accession": acc, "reason": "prediction is None"})
        else:
            pred = np.asarray(predictions[acc])
            if len(pred) == 0:
                failure_log.append({"accession": acc, "reason": "empty prediction array"})
            elif np.all(np.isnan(pred)):
                failure_log.append({"accession": acc, "reason": "all NaN predictions"})

    covered = total - len(failure_log)
    coverage_pct = (covered / total * 100) if total > 0 else 0.0

    return CoverageResult(
        total_proteins=total,
        covered_proteins=covered,
        coverage_pct=coverage_pct,
        failure_log=failure_log,
    )
