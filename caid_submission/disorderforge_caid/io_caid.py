"""FASTA input + CAID-format output (one .caid file per protein, timings.csv).

CAID output format (per residue, tab-separated):
    >P04637
    1\tM\t0.892\t1
    2\tE\t0.813\t1
    ...
column 1 = 1-based residue index, 2 = amino acid, 3 = disorder score [0,1],
column 4 = binary state at the method threshold.
"""
from __future__ import annotations

import csv
import os

# Ambiguous / non-standard FASTA characters CAID may include (must not crash):
#   B (N/D), Z (Q/E), J (L/I), U (Sec), O (Pyl), X (any). The lightweight
# feature table is 256-row ord-indexed and zero-fills unknown residues, and the
# pLM embeddings are precomputed upstream, so no special-casing is needed here —
# we only keep the raw character for the output column.
AMBIGUOUS = set("BZJUOX")


def read_fasta(path):
    """Yield (protein_id, sequence) in file order. Tolerates blank lines, wrapped
    sequences, and ambiguous residues. protein_id = first whitespace-delimited
    token after '>'."""
    pid, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if pid is not None:
                    yield pid, "".join(chunks)
                pid = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip().upper())
    if pid is not None:
        yield pid, "".join(chunks)


def write_caid(out_dir, protein_id, sequence, scores, binary):
    """Write out_dir/<protein_id>.caid. scores/binary are length-L sequences."""
    os.makedirs(out_dir, exist_ok=True)
    assert len(sequence) == len(scores) == len(binary), \
        f"{protein_id}: len mismatch seq={len(sequence)} scores={len(scores)} bin={len(binary)}"
    path = os.path.join(out_dir, f"{protein_id}.caid")
    with open(path, "w") as fh:
        fh.write(f">{protein_id}\n")
        for i, (aa, s, b) in enumerate(zip(sequence, scores, binary), start=1):
            fh.write(f"{i}\t{aa}\t{s:.4f}\t{int(b)}\n")
    return path


class Timings:
    """Accumulate per-sequence runtimes and write CAID timings.csv."""

    def __init__(self, path, header_note="DisorderForge-NOX"):
        self.path = path
        self.rows = []
        self.note = header_note

    def add(self, protein_id, milliseconds):
        self.rows.append((protein_id, int(round(milliseconds))))

    def write(self, started_str=None):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            if started_str:
                fh.write(f"# Running {self.note}, started {started_str}\n")
            w = csv.writer(fh)
            w.writerow(["sequence", "milliseconds"])
            for pid, ms in self.rows:
                w.writerow([pid, ms])
        return self.path
