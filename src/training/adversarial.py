"""Domain-adversarial training (P3-Ph2-S04).

Reference: P3_00 Amendment T-1; P3_04 §S04.

Implements:
  - GradientReversalLayer (GRL): identity in forward; negates and scales
    gradients (by lambda) in backward.
  - DomainDiscriminator: 2-layer MLP classifying residues as PDB-type
    (label 0) or NOX-type (label 1), trained from backbone penultimate
    features through the GRL.
  - lambda_schedule: linear ramp from `lambda_start` to `lambda_end` over
    training (per Amendment T-1: 0.01 → 1.0 over training epochs).

Combined training loss (handled in the training loop, GRL flips the sign):
    total = disorder_loss + lambda * domain_loss   (in code)
The GRL ensures the *backbone* receives  -lambda * d(domain_loss)/d(features)
while the discriminator itself receives the normal gradient — so the
backbone is pushed to make features domain-invariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Gradient Reversal Layer ─────────────────────────────────────────


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def gradient_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Apply the gradient-reversal layer with scale `lambda_`."""
    return _GradientReversalFunction.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """Module wrapper around `gradient_reverse`.

    Stores `lambda_` as an attribute — the training loop is expected to
    update it according to `lambda_schedule` between epochs.
    """

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reverse(x, self.lambda_)


# ─── Domain Discriminator ────────────────────────────────────────────


class DomainDiscriminator(nn.Module):
    """2-layer per-residue domain classifier (PDB-type vs NOX-type).

    Per the plan: Linear(hidden → hidden//2) → ReLU → Dropout(0.3) →
    Linear(hidden//2 → 2). Reads the backbone's penultimate features (after
    the GRL) and produces logits [B, L, 2].
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        mid = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Domain Loss ─────────────────────────────────────────────────────


def domain_adversarial_loss(
    discriminator_logits: torch.Tensor,   # [B, L, 2]
    domain_labels: torch.Tensor,          # [B] long: 0=PDB-type, 1=NOX-type, -1=UNKNOWN
    seq_mask: torch.Tensor | None = None, # [B, L] bool
) -> tuple[torch.Tensor, float, int]:
    """Cross-entropy domain loss over residues whose protein has a domain label.

    Returns (loss, accuracy, n_residues_used).

    accuracy is the discriminator's per-residue accuracy on the labeled subset
    (used for diagnostic logging — should drift toward 0.5 as the backbone
    learns domain-invariant features).
    """
    B, L, C = discriminator_logits.shape
    if seq_mask is None:
        seq_mask = torch.ones((B, L), dtype=torch.bool, device=discriminator_logits.device)
    seq_mask = seq_mask.bool()

    valid_proteins = (domain_labels != -1)  # [B]
    if int(valid_proteins.sum().item()) == 0:
        return (
            torch.tensor(0.0, device=discriminator_logits.device, requires_grad=True),
            0.5, 0,
        )

    # Broadcast per-protein domain labels to per-residue
    per_residue_labels = domain_labels.unsqueeze(1).expand(B, L)              # [B, L]
    per_residue_valid = valid_proteins.unsqueeze(1).expand(B, L) & seq_mask   # [B, L]

    flat_logits = discriminator_logits[per_residue_valid]                     # [N, 2]
    flat_labels = per_residue_labels[per_residue_valid].long()                # [N]
    if flat_logits.numel() == 0:
        return (
            torch.tensor(0.0, device=discriminator_logits.device, requires_grad=True),
            0.5, 0,
        )

    loss = F.cross_entropy(flat_logits, flat_labels)
    with torch.no_grad():
        preds = flat_logits.argmax(dim=-1)
        acc = (preds == flat_labels).float().mean().item()
    return loss, float(acc), int(flat_labels.numel())


# ─── Lambda Schedule ─────────────────────────────────────────────────


def lambda_schedule(
    epoch_idx: int,
    total_epochs: int,
    lambda_start: float = 0.01,
    lambda_end: float = 1.0,
    schedule: str = "linear",
) -> float:
    """Lambda value for a given epoch.

    Args:
        epoch_idx: 0-indexed current epoch.
        total_epochs: total epochs over which to ramp.
        lambda_start: lambda at epoch 0.
        lambda_end: lambda at epoch total_epochs - 1.
        schedule: "linear" (default) or "sigmoid" (slow start, fast middle).
    """
    if total_epochs <= 1:
        return lambda_end
    t = max(0.0, min(1.0, epoch_idx / (total_epochs - 1)))
    if schedule == "sigmoid":
        # Map t∈[0,1] through a sigmoid centered at 0.5 with steepness 10
        import math
        s = 1.0 / (1.0 + math.exp(-10 * (t - 0.5)))
        return float(lambda_start + (lambda_end - lambda_start) * s)
    # default: linear
    return float(lambda_start + (lambda_end - lambda_start) * t)
