"""Disorder function multi-task head (P4 Phase 2 lever EXP-E03).

Reference: P4_00 Amendment M-2 (DisoFLAG-style 6-category function multi-task).

Branches from an expert backbone's penultimate-layer features and predicts
six binary function annotations per residue (multi-label sigmoid):

    protein_binding, dna_binding, rna_binding,
    ion_binding, lipid_binding, flexible_linker

Loss: BCE per category, only on residues belonging to proteins that carry
ANY function annotation (per Phase 0 S04 — 765/3,409 = 22.4% coverage).
Residues in unannotated proteins are masked out entirely.

Combined training loss:
    total = disorder_loss + lambda_func * function_loss
Default lambda_func = 0.15 (per P4_04 §S01).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FUNCTION_CATEGORIES = [
    "protein_binding", "dna_binding", "rna_binding",
    "ion_binding", "lipid_binding", "flexible_linker",
]
NUM_FUNCTION_CATEGORIES = len(FUNCTION_CATEGORIES)


class FunctionMultiTaskHead(nn.Module):
    """Branches from backbone features → 6-class multi-label sigmoid per residue."""

    def __init__(self, hidden_dim: int, num_categories: int = NUM_FUNCTION_CATEGORIES,
                 dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_categories)
        self.drop = nn.Dropout(dropout)
        self.num_categories = num_categories

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, hidden_dim] → logits [B, L, num_categories]."""
        return self.linear(self.drop(hidden))


def function_multitask_loss(
    func_logits: torch.Tensor,      # [B, L, C]
    func_labels: torch.Tensor,      # [B, L, C] int8 — 1 = annotated, 0 = unannotated
    has_function: torch.Tensor,     # [B] bool — True iff protein has ANY function annotation
    seq_mask: torch.Tensor | None = None,  # [B, L] bool — real positions
) -> torch.Tensor:
    """BCE over residues belonging to function-annotated proteins only.

    Returns scalar mean over contributing (residue, category) pairs. Returns
    zero (preserving grad graph) if no protein in the batch carries function
    annotations.
    """
    B, L, C = func_logits.shape
    if has_function.sum() == 0:
        return func_logits.sum() * 0.0
    protein_mask = has_function.view(B, 1, 1).expand(B, L, C).to(func_logits.dtype)
    if seq_mask is not None:
        protein_mask = protein_mask * seq_mask.unsqueeze(-1).to(func_logits.dtype)
    targets = func_labels.to(func_logits.dtype)
    per_pos = F.binary_cross_entropy_with_logits(func_logits, targets, reduction="none")
    per_pos = per_pos * protein_mask
    denom = protein_mask.sum().clamp_min(1.0)
    return per_pos.sum() / denom
