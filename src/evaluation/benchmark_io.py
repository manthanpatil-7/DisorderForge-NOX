"""Benchmark I/O utilities for loading CAID benchmark proteins.

Reference: Phase 1 Plan P1-S06; Benchmark Contract §8.2

CAID reference file format (modified FASTA, 3 lines per protein):
    >DP03745
    MNASDFRRRGKEMVDYMADYLE...
    000011111000----------...

Annotation codes: 1=disordered, 0=ordered, -=masked/excluded
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"

# Label encoding consistent with Data Contract §9
LABEL_DISORDERED = 1
LABEL_ORDERED = 0
LABEL_MASKED = -1  # CAID '-' maps to masked (excluded from evaluation)

_ANNOTATION_MAP = {"1": LABEL_DISORDERED, "0": LABEL_ORDERED, "-": LABEL_MASKED}


@dataclass
class BenchmarkProtein:
    """A single protein from a CAID benchmark set."""

    accession: str  # DisProt accession (DP#####)
    sequence: str
    labels: list[int]  # per-residue: 1=disordered, 0=ordered, -1=masked
    length: int = field(init=False)

    def __post_init__(self):
        self.length = len(self.sequence)
        assert self.length == len(self.labels), (
            f"Sequence/label length mismatch for {self.accession}: "
            f"seq={self.length}, labels={len(self.labels)}"
        )
        assert self.length > 0, f"Empty protein: {self.accession}"

    @property
    def n_disordered(self) -> int:
        return sum(1 for la in self.labels if la == LABEL_DISORDERED)

    @property
    def n_ordered(self) -> int:
        return sum(1 for la in self.labels if la == LABEL_ORDERED)

    @property
    def n_masked(self) -> int:
        return sum(1 for la in self.labels if la == LABEL_MASKED)

    @property
    def has_both_classes(self) -> bool:
        """True if protein has at least one disordered AND one ordered residue."""
        return self.n_disordered > 0 and self.n_ordered > 0

    @property
    def is_fully_disordered(self) -> bool:
        """True if all assessed (non-masked) residues are disordered."""
        return self.n_ordered == 0 and self.n_disordered > 0

    @property
    def is_fully_ordered(self) -> bool:
        """True if all assessed (non-masked) residues are ordered."""
        return self.n_disordered == 0 and self.n_ordered > 0


def _parse_caid_fasta(path: Path) -> list[BenchmarkProtein]:
    """Parse a CAID reference FASTA file."""
    proteins = []
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        accession = lines[i][1:].strip().split()[0]
        sequence = lines[i + 1]
        annotation_str = lines[i + 2]
        labels = [_ANNOTATION_MAP[c] for c in annotation_str]
        proteins.append(BenchmarkProtein(
            accession=accession,
            sequence=sequence,
            labels=labels,
        ))
        i += 3

    return proteins


# Benchmark configuration: maps benchmark name to its files
_BENCHMARK_CONFIG = {
    "caid3": {
        "dir": "caid3",
        "subtracks": {
            "disorder_nox": "disorder_nox.fasta",
            "disorder_pdb": "disorder_pdb.fasta",
        },
        "use_dedup": False,
    },
    "caid2": {
        "dir": "caid2",
        "subtracks": {
            "disorder_nox": "disorder_nox_dedup.fasta",
            "disorder_pdb": "disorder_pdb_dedup.fasta",
        },
        "use_dedup": True,
    },
}


def load_benchmark(
    name: str,
    subtrack: Optional[str] = None,
) -> dict[str, BenchmarkProtein]:
    """Load a benchmark set, returning {accession: BenchmarkProtein}.

    Args:
        name: Benchmark name ("caid3" or "caid2").
        subtrack: Optional subtrack ("disorder_nox" or "disorder_pdb").
                  If None, loads all subtracks merged (later accessions
                  override earlier ones if duplicated, but within CAID
                  the same accession has different annotations per subtrack,
                  so callers should typically specify a subtrack).

    Returns:
        Dictionary mapping accession → BenchmarkProtein.
    """
    name_lower = name.lower()
    if name_lower not in _BENCHMARK_CONFIG:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {list(_BENCHMARK_CONFIG.keys())}"
        )

    config = _BENCHMARK_CONFIG[name_lower]
    bench_dir = BENCHMARKS_DIR / config["dir"]

    if subtrack is not None:
        if subtrack not in config["subtracks"]:
            raise ValueError(
                f"Unknown subtrack '{subtrack}' for {name}. "
                f"Available: {list(config['subtracks'].keys())}"
            )
        subtracks_to_load = {subtrack: config["subtracks"][subtrack]}
    else:
        subtracks_to_load = config["subtracks"]

    proteins: dict[str, BenchmarkProtein] = {}
    for st_name, fname in subtracks_to_load.items():
        fpath = bench_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Benchmark file not found: {fpath}")
        for p in _parse_caid_fasta(fpath):
            proteins[p.accession] = p

    return proteins


def load_benchmark_manifest(name: str) -> dict:
    """Load the manifest YAML for a benchmark."""
    name_lower = name.lower()
    if name_lower not in _BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark '{name}'.")
    config = _BENCHMARK_CONFIG[name_lower]
    manifest_path = BENCHMARKS_DIR / config["dir"] / "manifest.yaml"
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def get_benchmark_accessions(name: str) -> set[str]:
    """Get all accessions for a benchmark (across all subtracks)."""
    manifest = load_benchmark_manifest(name)
    accessions = set()
    for st in manifest["subtracks"].values():
        accessions.update(st["accessions"])
    return accessions
