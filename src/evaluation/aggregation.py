"""Per-protein macro-averaging with single-class protein exclusion.

Reference: Benchmark Contract §9.2

Protocol:
1. Compute metric per-protein (residue-level within each protein).
2. For AUC metrics (AUC-PR, AUC-ROC): exclude single-class proteins
   (fully disordered or fully ordered) — they have undefined AUC.
3. Macro-average: equal weight per protein (not length-weighted).
4. Report exclusion count and composition.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class MacroAverageResult:
    """Result of macro-averaging a metric across proteins."""

    mean: float
    std: float
    n_proteins: int  # number of proteins included in average
    excluded_count: int  # number of proteins excluded
    excluded_proportion: float  # excluded / total
    excluded_composition: dict  # {"fully_ordered": N, "fully_disordered": N}
    per_protein_values: list[float]  # individual protein scores (included only)
    excluded_accessions: list[str]  # accessions of excluded proteins


def compute_per_protein_metric(
    proteins: dict,
    predictions: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    exclude_single_class: bool = True,
    threshold: Optional[float] = None,
) -> MacroAverageResult:
    """Compute a metric per-protein and macro-average.

    Args:
        proteins: {accession: BenchmarkProtein} from benchmark_io.
        predictions: {accession: probability_array} predicted probabilities.
        metric_fn: Function(labels, probs) → float, or
                   Function(labels, probs, threshold) → float if threshold given.
        exclude_single_class: If True, exclude proteins with only one class
                              (required for AUC metrics).
        threshold: If provided, passed as third arg to metric_fn.

    Returns:
        MacroAverageResult with per-protein scores and exclusion info.
    """
    per_protein_values = []
    excluded_accessions = []
    excluded_fully_ordered = 0
    excluded_fully_disordered = 0

    for acc, protein in proteins.items():
        if acc not in predictions:
            continue

        probs = np.asarray(predictions[acc], dtype=float)
        # Filter to assessed residues (non-masked)
        labels = np.array(protein.labels)
        mask = labels >= 0  # 0=ordered, 1=disordered; -1=masked
        assessed_labels = labels[mask]
        assessed_probs = probs[mask]

        if len(assessed_labels) == 0:
            excluded_accessions.append(acc)
            continue

        # Check single-class
        unique_labels = set(assessed_labels.tolist())
        if exclude_single_class and len(unique_labels) < 2:
            excluded_accessions.append(acc)
            if unique_labels == {0}:
                excluded_fully_ordered += 1
            elif unique_labels == {1}:
                excluded_fully_disordered += 1
            continue

        # Compute metric
        if threshold is not None:
            score = metric_fn(assessed_labels, assessed_probs, threshold)
        else:
            score = metric_fn(assessed_labels, assessed_probs)

        per_protein_values.append(score)

    total = len([a for a in proteins if a in predictions])
    n_included = len(per_protein_values)
    n_excluded = total - n_included

    if n_included == 0:
        return MacroAverageResult(
            mean=float("nan"),
            std=float("nan"),
            n_proteins=0,
            excluded_count=n_excluded,
            excluded_proportion=1.0 if total > 0 else 0.0,
            excluded_composition={
                "fully_ordered": excluded_fully_ordered,
                "fully_disordered": excluded_fully_disordered,
            },
            per_protein_values=[],
            excluded_accessions=excluded_accessions,
        )

    values = np.array(per_protein_values)
    return MacroAverageResult(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        n_proteins=n_included,
        excluded_count=n_excluded,
        excluded_proportion=n_excluded / total if total > 0 else 0.0,
        excluded_composition={
            "fully_ordered": excluded_fully_ordered,
            "fully_disordered": excluded_fully_disordered,
        },
        per_protein_values=per_protein_values,
        excluded_accessions=excluded_accessions,
    )
