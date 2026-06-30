"""Late fusion adapter (P4 Phase 2 lever EXP-E06).

Reference: P3-00 M-4 (carried into P4); P4_04 §S01.

Concatenates the LIGHTWEIGHT physicochemical feature block (one-hot + simple
properties + positional, 41 dims by default) onto the backbone's penultimate
features just before the final linear head. The pLM embedding goes through
the deep stack; the lightweight features bypass the stack and join late.

  backbone(esm2 + lightweight) → hidden [B, L, hidden_dim]
                                        ↓ concat with raw lightweight [B, L, 41]
                                                 ↓ Linear → [B, L, 1]

This is architecture-agnostic — works with any backbone exposing
`forward_features()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LateFusionHead(nn.Module):
    """Replacement for the disorder output head when late fusion is enabled.

    Args:
        hidden_dim: dim of `hidden` from backbone.forward_features()
        physico_dim: dim of the lightweight features fed in via `physico` arg
                     (defaults to 41 = LIGHTWEIGHT_DIM).
        mlp_hidden: optional inner MLP dim; default = hidden_dim // 2.
        dropout: applied after concat.
    """

    def __init__(self, hidden_dim: int, physico_dim: int = 41,
                 mlp_hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = hidden_dim // 2
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + physico_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self.physico_dim = physico_dim

    def forward(self, hidden: torch.Tensor, physico: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, hidden_dim]; physico: [B, L, physico_dim] → [B, L, 1]"""
        return self.fuse(torch.cat([hidden, physico], dim=-1))
