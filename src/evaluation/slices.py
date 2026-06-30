"""Stress-slice assignment logic for benchmark proteins.

Reference: Benchmark Contract §12.3 (mandatory SS-1 through SS-6)
           Benchmark Contract §12.4 (diagnostic SS-7 through SS-9)

Slice membership is determined by ground-truth annotations, not predictions.
A protein may belong to multiple slices (slices are not mutually exclusive).
"""

from enum import Enum
from typing import Optional

from src.evaluation.benchmark_io import BenchmarkProtein


class SliceID(str, Enum):
    """Stress-slice identifiers."""
    # Mandatory (§12.3)
    SS1 = "SS-1"  # Short-IDR proteins (longest IDR ≤ 30)
    SS2 = "SS-2"  # Long-IDR proteins (longest IDR > 100)
    SS3 = "SS-3"  # Terminal-disorder-dominated (≥50% of disorder in first/last 30)
    SS4 = "SS-4"  # Internal-disorder-dominated (<10% of disorder in first/last 30)
    SS5 = "SS-5"  # Short proteins (length < 200)
    SS6 = "SS-6"  # Long proteins (length ≥ 500)
    # Diagnostic (§12.4)
    SS7 = "SS-7"  # Fully disordered (≥90% of assessed residues disordered)
    SS8 = "SS-8"  # Fully ordered (0% disordered among assessed)
    SS9 = "SS-9"  # Boundary-heavy (≥5 order-disorder transitions)


def _longest_contiguous_disordered(labels: list[int]) -> int:
    """Length of the longest contiguous run of disordered (1) residues."""
    max_run = 0
    current_run = 0
    for la in labels:
        if la == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _terminal_disorder_count(labels: list[int], terminal_size: int = 30) -> int:
    """Count disordered residues within first or last `terminal_size` positions."""
    n = len(labels)
    count = 0
    for i, la in enumerate(labels):
        if la == 1 and (i < terminal_size or i >= n - terminal_size):
            count += 1
    return count


def _count_transitions(labels: list[int]) -> int:
    """Count order-disorder transitions (boundary crossings).

    Counts transitions between adjacent assessed (non-masked) residues.
    Masked residues are skipped (no transition counted across them).
    """
    assessed = [(i, la) for i, la in enumerate(labels) if la >= 0]
    transitions = 0
    for j in range(1, len(assessed)):
        if assessed[j][1] != assessed[j - 1][1]:
            transitions += 1
    return transitions


def assign_slices(protein: BenchmarkProtein) -> set[SliceID]:
    """Assign a protein to all applicable stress slices.

    Uses ground-truth annotations only.

    Args:
        protein: BenchmarkProtein with labels.

    Returns:
        Set of SliceIDs this protein belongs to.
    """
    slices: set[SliceID] = set()
    labels = protein.labels
    length = protein.length
    n_dis = protein.n_disordered
    n_ord = protein.n_ordered
    assessed = n_dis + n_ord

    # SS-1: Short-IDR proteins (longest contiguous IDR ≤ 30)
    longest_idr = _longest_contiguous_disordered(labels)
    if n_dis > 0 and longest_idr <= 30:
        slices.add(SliceID.SS1)

    # SS-2: Long-IDR proteins (longest contiguous IDR > 100)
    if longest_idr > 100:
        slices.add(SliceID.SS2)

    # SS-3: Terminal-disorder-dominated (≥50% of disordered in first/last 30)
    if n_dis > 0:
        terminal_dis = _terminal_disorder_count(labels, terminal_size=30)
        if terminal_dis / n_dis >= 0.50:
            slices.add(SliceID.SS3)

        # SS-4: Internal-disorder-dominated (<10% of disordered in first/last 30)
        if terminal_dis / n_dis < 0.10:
            slices.add(SliceID.SS4)

    # SS-5: Short proteins (length < 200)
    if length < 200:
        slices.add(SliceID.SS5)

    # SS-6: Long proteins (length ≥ 500)
    if length >= 500:
        slices.add(SliceID.SS6)

    # SS-7: Fully disordered (≥90% of assessed residues are disordered)
    if assessed > 0 and n_dis / assessed >= 0.90:
        slices.add(SliceID.SS7)

    # SS-8: Fully ordered (0% disordered among assessed)
    if assessed > 0 and n_dis == 0:
        slices.add(SliceID.SS8)

    # SS-9: Boundary-heavy (≥5 order-disorder transitions)
    if _count_transitions(labels) >= 5:
        slices.add(SliceID.SS9)

    return slices


def slice_benchmark(
    benchmark: dict[str, BenchmarkProtein],
    slice_id: SliceID,
) -> list[BenchmarkProtein]:
    """Filter benchmark proteins to those belonging to a given slice.

    Args:
        benchmark: {accession: BenchmarkProtein}.
        slice_id: Which slice to filter by.

    Returns:
        List of BenchmarkProtein in the slice.
    """
    return [p for p in benchmark.values() if slice_id in assign_slices(p)]
