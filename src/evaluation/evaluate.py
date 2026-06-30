"""Top-level evaluation orchestrator.

Reference: Phase 1 Plan P1-S13; Benchmark Contract §16.4

Ties together metrics, aggregation, slices, coverage, CIs, and threshold
into a structured EvaluationReport. This is the report schema consumed by
Phase 4 (validation monitoring) and Phase 5 (final benchmark evaluation).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import yaml

from src.evaluation.aggregation import MacroAverageResult, compute_per_protein_metric
from src.evaluation.benchmark_io import BenchmarkProtein, load_benchmark
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.coverage import CoverageResult, compute_coverage
from src.evaluation.metrics import (
    auc_pr,
    auc_roc,
    balanced_accuracy_at_threshold,
    fmax,
    mcc_at_threshold,
)
from src.evaluation.slices import SliceID, assign_slices
from src.evaluation.threshold import compute_metrics_at_threshold


@dataclass
class SliceReport:
    """Metrics for a single stress slice."""

    slice_id: str
    n_proteins: int
    p1_mean: Optional[float] = None
    p1_std: Optional[float] = None
    p2_mean: Optional[float] = None
    p2_std: Optional[float] = None
    fmax_mean: Optional[float] = None
    excluded_count: int = 0
    underpowered: bool = False  # True if <5 proteins


@dataclass
class EvaluationReport:
    """Structured evaluation report per Benchmark Contract §16.4.

    Contains all fields required for Phase 4 validation monitoring
    and Phase 5 final benchmark evaluation.
    """

    benchmark_name: str
    subtrack: Optional[str]

    # Primary metrics (P1, P2) — macro-averaged
    p1_mean: float  # AUC-PR
    p1_std: float
    p2_mean: float  # AUC-ROC
    p2_std: float

    # Diagnostic metrics
    fmax_mean: float
    mcc_mean: Optional[float] = None  # requires frozen threshold
    balanced_accuracy_mean: Optional[float] = None
    frozen_threshold: Optional[float] = None

    # Coverage
    coverage_pct: float = 0.0
    coverage_total: int = 0
    coverage_covered: int = 0

    # Exclusion reporting (Benchmark Contract §9.2)
    p1_excluded_count: int = 0
    p1_excluded_proportion: float = 0.0
    p1_excluded_composition: dict = field(default_factory=dict)
    p2_excluded_count: int = 0
    p2_excluded_proportion: float = 0.0

    # Bootstrap CIs (Benchmark Contract §10.2)
    p1_ci_lower: Optional[float] = None
    p1_ci_upper: Optional[float] = None
    p2_ci_lower: Optional[float] = None
    p2_ci_upper: Optional[float] = None

    # Per-slice reports (Benchmark Contract §12.5)
    slice_reports: list = field(default_factory=list)

    # Per-protein values for downstream analysis
    per_protein_p1: list = field(default_factory=list)
    per_protein_p2: list = field(default_factory=list)

    def validate(self):
        """Assert no required field is None after construction."""
        assert self.benchmark_name, "benchmark_name is empty"
        assert not np.isnan(self.p1_mean) or self.p1_excluded_count > 0, "P1 is NaN"
        assert not np.isnan(self.p2_mean) or self.p2_excluded_count > 0, "P2 is NaN"
        assert 0.0 <= self.coverage_pct <= 100.0, f"Invalid coverage: {self.coverage_pct}"

    def to_dict(self) -> dict:
        """Serialize to dict (JSON/YAML compatible)."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Serialize to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str) -> None:
        """Serialize to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationReport":
        """Deserialize from dict."""
        return cls(**d)


def evaluate_on_benchmark(
    predictions: dict[str, np.ndarray],
    benchmark_name: str,
    subtrack: Optional[str] = None,
    frozen_threshold: Optional[float] = None,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: Optional[int] = None,
) -> EvaluationReport:
    """Run full evaluation pipeline on a benchmark.

    Args:
        predictions: {accession: probability_array}.
        benchmark_name: "caid3" or "caid2".
        subtrack: Optional subtrack name.
        frozen_threshold: If provided, compute MCC/BalAcc at this threshold.
        bootstrap_resamples: Number of bootstrap resamples for CIs.
        bootstrap_seed: Optional seed for reproducible CIs.

    Returns:
        EvaluationReport with all required fields.
    """
    proteins = load_benchmark(benchmark_name, subtrack=subtrack)

    # Coverage
    cov = compute_coverage(proteins, predictions)

    # P1: macro-averaged AUC-PR (with single-class exclusion)
    p1_result = compute_per_protein_metric(
        proteins, predictions, auc_pr, exclude_single_class=True
    )

    # P2: macro-averaged AUC-ROC (with single-class exclusion)
    p2_result = compute_per_protein_metric(
        proteins, predictions, auc_roc, exclude_single_class=True
    )

    # Fmax (no single-class exclusion — handles all-positive with 0)
    fmax_result = compute_per_protein_metric(
        proteins, predictions, fmax, exclude_single_class=False
    )

    # Threshold-dependent diagnostics
    mcc_mean = None
    balacc_mean = None
    if frozen_threshold is not None:
        mcc_result = compute_per_protein_metric(
            proteins, predictions, mcc_at_threshold,
            exclude_single_class=False, threshold=frozen_threshold,
        )
        balacc_result = compute_per_protein_metric(
            proteins, predictions, balanced_accuracy_at_threshold,
            exclude_single_class=False, threshold=frozen_threshold,
        )
        mcc_mean = mcc_result.mean
        balacc_mean = balacc_result.mean

    # Bootstrap CIs
    p1_ci = bootstrap_ci(
        p1_result.per_protein_values,
        n_resamples=bootstrap_resamples,
        rng_seed=bootstrap_seed,
    )
    p2_ci = bootstrap_ci(
        p2_result.per_protein_values,
        n_resamples=bootstrap_resamples,
        rng_seed=bootstrap_seed,
    )

    # Per-slice evaluation (mandatory slices)
    slice_reports = []
    mandatory_slices = [SliceID.SS1, SliceID.SS2, SliceID.SS3,
                        SliceID.SS4, SliceID.SS5, SliceID.SS6]

    # Pre-compute slice membership
    protein_slices = {acc: assign_slices(p) for acc, p in proteins.items()}

    for sid in mandatory_slices:
        slice_proteins = {
            acc: p for acc, p in proteins.items()
            if sid in protein_slices.get(acc, set())
        }
        n_in_slice = len(slice_proteins)

        if n_in_slice == 0:
            slice_reports.append(SliceReport(
                slice_id=sid.value, n_proteins=0, underpowered=True
            ))
            continue

        sp1 = compute_per_protein_metric(
            slice_proteins, predictions, auc_pr, exclude_single_class=True
        )
        sp2 = compute_per_protein_metric(
            slice_proteins, predictions, auc_roc, exclude_single_class=True
        )
        sfmax = compute_per_protein_metric(
            slice_proteins, predictions, fmax, exclude_single_class=False
        )

        slice_reports.append(SliceReport(
            slice_id=sid.value,
            n_proteins=n_in_slice,
            p1_mean=sp1.mean if sp1.n_proteins > 0 else None,
            p1_std=sp1.std if sp1.n_proteins > 0 else None,
            p2_mean=sp2.mean if sp2.n_proteins > 0 else None,
            p2_std=sp2.std if sp2.n_proteins > 0 else None,
            fmax_mean=sfmax.mean if sfmax.n_proteins > 0 else None,
            excluded_count=sp1.excluded_count,
            underpowered=n_in_slice < 5,
        ))

    report = EvaluationReport(
        benchmark_name=benchmark_name,
        subtrack=subtrack,
        p1_mean=p1_result.mean,
        p1_std=p1_result.std,
        p2_mean=p2_result.mean,
        p2_std=p2_result.std,
        fmax_mean=fmax_result.mean,
        mcc_mean=mcc_mean,
        balanced_accuracy_mean=balacc_mean,
        frozen_threshold=frozen_threshold,
        coverage_pct=cov.coverage_pct,
        coverage_total=cov.total_proteins,
        coverage_covered=cov.covered_proteins,
        p1_excluded_count=p1_result.excluded_count,
        p1_excluded_proportion=p1_result.excluded_proportion,
        p1_excluded_composition=p1_result.excluded_composition,
        p2_excluded_count=p2_result.excluded_count,
        p2_excluded_proportion=p2_result.excluded_proportion,
        p1_ci_lower=p1_ci[0],
        p1_ci_upper=p1_ci[1],
        p2_ci_lower=p2_ci[0],
        p2_ci_upper=p2_ci[1],
        slice_reports=[asdict(sr) for sr in slice_reports],
        per_protein_p1=p1_result.per_protein_values,
        per_protein_p2=p2_result.per_protein_values,
    )

    report.validate()
    return report
