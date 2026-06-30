"""Mixture of Experts head (Part 5 Phase 3 EXP-A01).

Reference: P5_05 §P5-Ph3-S03; CLAUDE.md rule 32 (LOAD-BALANCED gating).

Mechanism:
  - K expert heads, each Linear(hidden_dim → 1) producing logits.
  - Gating network: Linear(hidden_dim → K) → softmax over K.
  - Output logit per residue = sum_k gate_k * expert_k.
  - Load-balancing auxiliary loss: penalizes unbalanced expert utilization.
    Switch-Transformer-style: importance × frequency loss.

Per CLAUDE.md gotcha: monitor per-expert utilization every epoch; if any
expert handles <10% of residues after epoch 5, increase load-balance weight.

The MoEHead replaces CNNTransformerHybrid.output_head. The training loop must
add the load_balance_loss to the total loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEHead(nn.Module):
    """K-expert mixture for binary disorder classification.

    Forward returns:
        logit:  [B, L, 1]    — gate-weighted soft prediction (logit space)
        gate:   [B, L, K]    — softmax gate weights (for diagnostics)
        per_expert_logit: [B, L, K]  — raw expert logits (for analysis)
    """

    def __init__(self, hidden_dim: int, num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"num_experts must be >= 2; got {num_experts}")
        self.K = num_experts
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, h: torch.Tensor) -> dict:
        """h: [B, L, hidden_dim] → dict with logit/gate/per_expert_logit."""
        h = self.dropout(h)
        # Per-expert logits: [B, L, K]
        per_expert = torch.stack([e(h).squeeze(-1) for e in self.experts], dim=-1)
        # Gating: [B, L, K]
        gate_logits = self.gate(h)
        gate_probs = F.softmax(gate_logits, dim=-1)
        # Combine: gate-weighted sum → [B, L]
        logit = (gate_probs * per_expert).sum(dim=-1, keepdim=True)
        return {
            "logit": logit,                       # [B, L, 1]
            "gate": gate_probs,                   # [B, L, K]
            "per_expert_logit": per_expert,       # [B, L, K]
            "gate_logits": gate_logits,           # [B, L, K] (pre-softmax)
        }


def load_balancing_loss(
    gate_probs: torch.Tensor,    # [B, L, K]
    seq_mask: torch.Tensor,       # [B, L] bool — True = valid residue
) -> torch.Tensor:
    """Switch-Transformer style auxiliary loss.

    For each expert k:
        importance_k = mean over valid residues of gate_probs[..., k]
        frequency_k  = mean over valid residues of (argmax_dim_K == k)

    Loss = K * sum_k(importance_k * frequency_k)

    At perfect uniform routing, importance_k = frequency_k = 1/K and the loss
    equals 1.0. Higher means imbalance (some experts get more attention than
    they're chosen, or vice versa).
    """
    K = gate_probs.size(-1)
    mask = seq_mask.float().unsqueeze(-1)                    # [B, L, 1]
    denom = mask.sum().clamp_min(1.0)

    importance = (gate_probs * mask).sum(dim=(0, 1)) / denom   # [K]

    chosen = gate_probs.argmax(dim=-1)                          # [B, L]
    one_hot = F.one_hot(chosen, num_classes=K).float()          # [B, L, K]
    frequency = (one_hot * mask).sum(dim=(0, 1)) / denom        # [K]

    return K * (importance * frequency).sum()


def expert_utilization(
    gate_probs: torch.Tensor,    # [B, L, K]
    seq_mask: torch.Tensor,       # [B, L] bool
    method: str = "argmax",
) -> torch.Tensor:
    """Per-expert utilization (fraction of residues routed to each expert).

    method:
        "argmax" — fraction of residues whose hard argmax is k
        "mean_prob" — mean gate probability (matches the load-balance importance)
    """
    K = gate_probs.size(-1)
    mask = seq_mask.float().unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    if method == "argmax":
        chosen = gate_probs.argmax(dim=-1)
        one_hot = F.one_hot(chosen, num_classes=K).float()
        return (one_hot * mask).sum(dim=(0, 1)) / denom
    return (gate_probs * mask).sum(dim=(0, 1)) / denom


class MoEHybridWrapper(nn.Module):
    """Wraps CNNTransformerHybrid backbone + MoE head.

    The wrapper exposes the same interface the trainer expects (forward(x,
    key_padding_mask=...) → logit) plus an optional .moe_diag attribute set
    on the last forward call for the loss calculation.
    """

    def __init__(self, backbone: nn.Module, num_experts: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.moe = MoEHead(
            hidden_dim=backbone.working_dim,
            num_experts=num_experts, dropout=dropout,
        )
        self._last_diag = None

    @property
    def working_dim(self):
        return self.backbone.working_dim

    @property
    def output_head(self):
        # For backward compatibility (some scripts access .output_head)
        return self.moe

    def forward_features(self, x, key_padding_mask=None):
        return self.backbone.forward_features(x, key_padding_mask=key_padding_mask)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        h = self.backbone.forward_features(x, key_padding_mask=key_padding_mask)
        out = self.moe(h)
        self._last_diag = out
        return out["logit"]

    def last_gate(self) -> torch.Tensor | None:
        return None if self._last_diag is None else self._last_diag["gate"]

    def last_per_expert_logit(self) -> torch.Tensor | None:
        return None if self._last_diag is None else self._last_diag["per_expert_logit"]
