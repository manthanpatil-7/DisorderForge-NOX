"""Protein-level context injection (P4 Phase 2 lever EXP-E07).

Reference: P4_00 Amendment M-3.

Computes 4 global features per protein and broadcasts each to every residue,
then concatenates with the backbone's penultimate features before the final
classification linear. Concretely the four features are:

  1. mean_predicted_disorder  — mean of Part 3 ensemble predictions over the
                                 protein (single scalar). At training time
                                 this is precomputed; at inference the same
                                 cached prediction is used.
  2. log_length               — log(L) / log(L_max_train)  (~unit scale)
  3. aa_composition_bias      — Shannon-entropy-normalized AA frequency
                                 against canonical-20 uniform: a single scalar
                                 in [0, 1] where 1 = uniform composition.
  4. net_charge_per_residue   — (count(K)+count(R) − count(D)−count(E)) / L
                                 (single scalar in roughly [-1, 1]).

`ProteinContextEncoder` is architecture-agnostic — it concatenates a 4-dim
broadcast onto a `[B, L, hidden_dim]` feature tensor and passes the result
through a small MLP back to `[B, L, hidden_dim]` for the existing output
head. Drop-in: backbone produces `hidden`, encoder consumes (hidden, ctx),
output head sees the same shape it always did.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Canonical 20 AA used for composition computation
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {a: i for i, a in enumerate(CANONICAL_AA)}
POSITIVE = set("KR")
NEGATIVE = set("DE")

NUM_PROTEIN_CONTEXT_FEATURES = 4


def compute_protein_context(
    sequence: str,
    mean_predicted_disorder: float,
    L_max_train: int = 5000,
) -> np.ndarray:
    """Returns shape [4] of float32. Computed once per protein."""
    L = max(1, len(sequence))
    seq_upper = sequence.upper()

    # 1. mean predicted disorder — passed in (precomputed from Part 3 ensemble)
    f_mean_disorder = float(mean_predicted_disorder)

    # 2. log length, normalized by log(L_max_train)
    f_log_len = math.log(L) / math.log(max(2, L_max_train))

    # 3. AA composition uniformity (Shannon H over canonical 20, normalized)
    counts = np.zeros(20, dtype=np.float64)
    n_canonical = 0
    for a in seq_upper:
        idx = AA_TO_INDEX.get(a, -1)
        if idx >= 0:
            counts[idx] += 1
            n_canonical += 1
    if n_canonical > 0:
        p = counts / n_canonical
        nonzero = p > 0
        H = float(-np.sum(p[nonzero] * np.log2(p[nonzero])))
        H_max = math.log2(20)
        f_aa_uniformity = H / H_max
    else:
        f_aa_uniformity = 0.0

    # 4. Net charge per residue
    npos = sum(1 for a in seq_upper if a in POSITIVE)
    nneg = sum(1 for a in seq_upper if a in NEGATIVE)
    f_net_charge = (npos - nneg) / L

    return np.array(
        [f_mean_disorder, f_log_len, f_aa_uniformity, f_net_charge],
        dtype=np.float32,
    )


class ProteinContextEncoder(nn.Module):
    """Concat per-residue hidden features with broadcast protein context,
    then MLP back to hidden_dim. Drop-in between backbone.forward_features()
    and the disorder output head."""

    def __init__(self, hidden_dim: int, context_dim: int = NUM_PROTEIN_CONTEXT_FEATURES,
                 mlp_hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.context_dim = context_dim

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, hidden_dim]; context: [B, context_dim] → [B, L, hidden_dim]"""
        B, L, _ = hidden.shape
        broadcast = context.view(B, 1, self.context_dim).expand(B, L, self.context_dim)
        combined = torch.cat([hidden, broadcast], dim=-1)
        return self.mlp(combined)
