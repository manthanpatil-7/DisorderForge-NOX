"""Differentiable Average-Precision loss (Lever 17, Part 6).

Direct surrogate for per-protein rpAP. Standard BCE optimizes the wrong
objective — at the residue level, the eval metric is rpAP (average
precision averaged across proteins), and there is a known gap between
BCE-optimal and AP-optimal score distributions.

Soft-AP formulation:
  For one protein with per-residue scores s ∈ R^L and labels y ∈ {0, 1}^L,
  let τ > 0 be a temperature. For each positive index i:
      soft_rank(i) = 1 + Σ_{j ≠ i} σ((s_j - s_i) / τ)
      soft_TP_at_i = 1 + Σ_{j ≠ i} σ((s_j - s_i) / τ) · y_j
      soft_precision(i) = soft_TP_at_i / soft_rank(i)
  Soft-AP for the protein:
      sAP = (1 / n_pos) Σ_{i: y_i = 1} soft_precision(i)
  Loss = 1 - sAP.

As τ → 0, σ → step indicator and sAP → exact AP. We use τ = 0.1 by
default (Rule 51). Smaller τ gives a sharper surrogate but unstable
gradients near ties; larger τ is smoother but biases toward BCE.

Schedule (Rule 51):
  β = 0   for first 50% of epochs   (pure BCE)
  β = 0.3 for last 50% of epochs    (mixed: 0.7 * BCE + 0.3 * DiffAP)
  If divergence is observed: reduce β or increase τ.

Masking:
  Per-residue mask m ∈ {0, 1}^L (1 = valid). Invalid residues are
  excluded from BOTH the rank summation and the positive-index average.

Per-protein vs. pooled:
  rpAP is per-protein. We compute the soft surrogate INDEPENDENTLY per
  protein in the batch, then average across proteins. Pooled
  computation (concatenate-then-AP) would optimize the wrong metric.

Reference:
  Brown et al., "Smooth-AP: smoothing the path towards large-scale
  image retrieval." ECCV 2020. (Same surrogate, applied to embedding
  retrieval; we apply it to per-residue binary classification.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_ap_per_protein(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Soft average precision for ONE protein.

    Args:
        scores: (L,) raw logit OR sigmoid-output; rank invariant only depends
            on differences, so use whichever is more stable. Logits recommended.
        labels: (L,) {0, 1} int or float
        mask:   (L,) bool or float, 1 where valid
        tau:    sigmoid temperature

    Returns:
        scalar in [0, 1]. NaN-safe: returns 1.0 if no positives.
    """
    valid = mask.bool()
    s = scores[valid]
    y = labels[valid].float()
    n = s.numel()
    if n == 0:
        return scores.new_tensor(1.0)
    pos_count = y.sum()
    if pos_count.item() == 0:
        return scores.new_tensor(1.0)

    # Pairwise score differences  D[i, j] = s_j - s_i
    diffs = s.unsqueeze(0) - s.unsqueeze(1)
    soft_indicator = torch.sigmoid(diffs / tau)
    # Zero the diagonal so j ≠ i.
    eye = torch.eye(n, device=s.device, dtype=soft_indicator.dtype)
    soft_indicator = soft_indicator * (1.0 - eye)

    # soft_rank[i]      = 1 + Σ_j σ((s_j - s_i)/τ)
    # soft_TP_at_i      = 1 + Σ_j σ((s_j - s_i)/τ) · y_j      (i contributes its own y_i = 1
    #                                                          for positive i because the
    #                                                          rank-1 numerator always
    #                                                          credits itself)
    soft_rank = 1.0 + soft_indicator.sum(dim=1)
    soft_tp = 1.0 + (soft_indicator * y.unsqueeze(0)).sum(dim=1)

    precision = soft_tp / soft_rank
    sap = (precision * y).sum() / pos_count
    return sap


def diff_ap_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Batched differentiable AP loss.

    Args:
        scores: (B, L) logits OR probabilities
        labels: (B, L) {0, 1, -1}; -1 (or mask=0) marks ignored positions
        mask:   (B, L) bool or float; 1 = valid. Overrides label==-1.
        tau:    sigmoid temperature (Rule 51 default 0.1)

    Returns:
        scalar = 1 - mean over proteins (with ≥1 positive) of soft-AP.
        Proteins with 0 positives are excluded from the average.
    """
    B = scores.shape[0]
    aps = []
    for b in range(B):
        m_b = mask[b].float() * (labels[b] != -1).float()
        sap = soft_ap_per_protein(scores[b], labels[b].clamp(min=0).float(),
                                  m_b, tau)
        if (labels[b].clamp(min=0).float() * m_b).sum() > 0:
            aps.append(sap)
    if not aps:
        return scores.new_tensor(0.0, requires_grad=True)
    return 1.0 - torch.stack(aps).mean()


class DiffAPMixedLoss(nn.Module):
    """BCE + scheduled DiffAP loss (Rule 51).

    Args:
        tau:        temperature (default 0.1)
        beta_max:   final DiffAP weight (default 0.3)
        warmup_frac: fraction of total epochs with β = 0 (default 0.5)

    Usage:
        loss_fn = DiffAPMixedLoss(tau=0.1, beta_max=0.3, warmup_frac=0.5)
        # in training loop:
        loss_fn.set_epoch(epoch, total_epochs)
        loss = loss_fn(logits, labels, mask)

    The schedule is a step function (β jumps from 0 → β_max at warmup_frac).
    A smoother ramp is possible but Rule 51 specifies the step schedule.
    """

    def __init__(
        self,
        tau: float = 0.1,
        beta_max: float = 0.3,
        warmup_frac: float = 0.5,
        bce_pos_weight: float | None = None,
    ):
        super().__init__()
        if not 0 <= beta_max <= 1:
            raise ValueError(f"beta_max must be in [0, 1], got {beta_max}")
        if not 0 <= warmup_frac <= 1:
            raise ValueError(f"warmup_frac must be in [0, 1], got {warmup_frac}")
        self.tau = tau
        self.beta_max = beta_max
        self.warmup_frac = warmup_frac
        self.register_buffer(
            "pos_weight",
            torch.tensor(bce_pos_weight) if bce_pos_weight is not None
            else torch.tensor(1.0),
        )
        self._beta = 0.0

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Call at the start of each epoch to update the schedule."""
        if total_epochs <= 0:
            self._beta = self.beta_max
            return
        frac_done = epoch / total_epochs
        self._beta = self.beta_max if frac_done >= self.warmup_frac else 0.0

    @property
    def beta(self) -> float:
        return self._beta

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """logits: (B, L), labels: (B, L) ∈ {-1, 0, 1}, mask: (B, L) bool/float."""
        valid = (mask.float() * (labels != -1).float()).bool()
        # BCE component — clip targets to {0, 1} via .clamp(min=0)
        bce = F.binary_cross_entropy_with_logits(
            logits, labels.clamp(min=0).float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        bce = (bce * valid.float()).sum() / valid.float().sum().clamp(min=1)

        if self._beta == 0.0:
            return bce

        # DiffAP — operates on logits directly (rank-invariant).
        ap_loss = diff_ap_loss(logits, labels, valid.float(), tau=self.tau)
        return (1.0 - self._beta) * bce + self._beta * ap_loss


__all__ = ["soft_ap_per_protein", "diff_ap_loss", "DiffAPMixedLoss"]
