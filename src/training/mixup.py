"""Mixup data augmentation on protein embeddings (Part 5 Phase 2 EXP-T02).

Reference: P5_04 §P5-Ph2-S06; CLAUDE.md gotcha (mixup between different lengths).

Mechanism:
  - Pair proteins of similar length (within 20%, truncate longer to shorter).
  - Sample lambda ~ Beta(alpha, alpha) with alpha=0.2 (heavy U-shape).
  - Interpolate embeddings:        emb_mix = lam * A + (1-lam) * B
  - Interpolate labels (continuous): lab_mix = lam * y_A + (1-lam) * y_B
  - Apply BCE on continuous mixed labels.
  - Mask: if either A or B at residue i is masked/ambiguous, the residue is
    excluded from the loss (we treat the residue as "uncertain" rather than
    half-missing).

Used on 50% of batches; the remaining 50% use unmixed batches so the model
still sees real data. The training loop chooses per-batch which mode to use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Label constants
DISORDERED = 1
ORDERED = 0
MASKED = -1
AMBIGUOUS = -2


def beta_lambda(alpha: float = 0.2, device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample lambda ~ Beta(alpha, alpha)."""
    if alpha <= 0:
        return torch.tensor(1.0, device=device)
    a = torch.tensor([alpha], device=device)
    b = torch.tensor([alpha], device=device)
    dist = torch.distributions.Beta(a, b)
    lam = dist.sample().to(device).squeeze()
    return lam


def length_compatible_pairs(
    lengths: torch.Tensor,   # [B]
    tolerance: float = 0.2,
) -> torch.Tensor:
    """Return a permutation [B] mapping i -> j such that |L_i - L_j| / max(L_i, L_j) <= tolerance.

    Greedy: for each i pick a random j that satisfies the tolerance and is
    not already used. Falls back to identity (no-mix) if no compatible
    partner exists for some i.
    """
    B = lengths.size(0)
    perm = torch.arange(B, device=lengths.device)
    used = torch.zeros(B, dtype=torch.bool, device=lengths.device)
    order = torch.randperm(B, device=lengths.device)
    for i in order.tolist():
        if used[i]:
            continue
        Li = lengths[i].item()
        # find j != i, not used, length compatible
        candidates = []
        for j in range(B):
            if j == i or used[j]:
                continue
            Lj = lengths[j].item()
            if max(Li, Lj) == 0:
                continue
            if abs(Li - Lj) / max(Li, Lj) <= tolerance:
                candidates.append(j)
        if not candidates:
            continue
        j = candidates[torch.randint(len(candidates), (1,)).item()]
        perm[i] = j
        perm[j] = i
        used[i] = True
        used[j] = True
    return perm


def mix_features_and_labels(
    features: torch.Tensor,    # [B, L, F]
    labels: torch.Tensor,       # [B, L]
    lengths: torch.Tensor,      # [B]
    alpha: float = 0.2,
    length_tolerance: float = 0.2,
) -> dict:
    """Apply mixup to a batch in-place style. Returns mixed features, soft
    labels, mask, and the lambda used.

    For paired (i, j): truncate both to min(L_i, L_j), mix.
    For unpaired i: lambda=1 (no mix), keep original.
    """
    B, L, _ = features.shape
    device = features.device

    perm = length_compatible_pairs(lengths, tolerance=length_tolerance)
    lam = beta_lambda(alpha=alpha, device=device)
    # Symmetric mixup: lam should be in [0.5, 1] to avoid identical (i,j)/(j,i) results
    lam = torch.where(lam < 0.5, 1 - lam, lam)

    mixed_feats = features.clone()
    mixed_targets = labels.float().clone()  # Will be soft
    valid_mask = torch.zeros((B, L), dtype=torch.float32, device=device)

    is_disordered = (labels == DISORDERED)
    is_ordered = (labels == ORDERED)
    is_supervised = is_disordered | is_ordered

    for i in range(B):
        j = perm[i].item()
        if j == i:
            # No partner -> keep original; valid mask = original supervised mask
            mixed_targets[i] = is_disordered[i].float()
            valid_mask[i] = is_supervised[i].float()
            continue
        Li = lengths[i].item()
        Lj = lengths[j].item()
        Lmin = min(Li, Lj)

        feat_i = features[i, :Lmin]
        feat_j = features[j, :Lmin]
        mixed_feats[i, :Lmin] = lam * feat_i + (1.0 - lam) * feat_j
        mixed_feats[i, Lmin:] = 0.0

        # Soft target: only valid where BOTH residues supervised
        sup_i = is_supervised[i, :Lmin].float()
        sup_j = is_supervised[j, :Lmin].float()
        both = sup_i * sup_j  # [Lmin]

        y_i = is_disordered[i, :Lmin].float()
        y_j = is_disordered[j, :Lmin].float()
        soft = lam * y_i + (1.0 - lam) * y_j  # [Lmin]

        mixed_targets[i, :Lmin] = soft
        mixed_targets[i, Lmin:] = 0.0
        valid_mask[i, :Lmin] = both
        valid_mask[i, Lmin:] = 0.0

    return {
        "features": mixed_feats,
        "soft_targets": mixed_targets,  # [B, L] continuous in [0, 1]
        "valid_mask": valid_mask,        # [B, L] 0/1 — residues with valid soft target
        "lam": lam.item(),
        "perm": perm,
    }


def mixup_bce_loss(
    logits: torch.Tensor,        # [B, L, 1] or [B, L]
    soft_targets: torch.Tensor,  # [B, L] continuous
    valid_mask: torch.Tensor,    # [B, L]
) -> torch.Tensor:
    """BCE on soft targets, masked to valid positions."""
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    per = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
    per = per * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    loss = per.sum() / denom
    return loss
