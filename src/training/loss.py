"""Masked losses for disorder prediction.

Reference: Training Contract §8.1–§8.3; P2_00 Amendment T-1 (focal loss),
Amendment T-4 (tiered weighting).

Label scheme (Data Contract §9):
  DISORDERED = +1  → target = 1.0
  ORDERED    =  0  → target = 0.0
  AMBIGUOUS  = -2  → excluded (zero gradient)
  MASKED     = -1  → excluded (zero gradient)

Loss is averaged over contributing residues only (not total residues).
Per-protein tier weighting (Amendment T-4) is applied as a multiplier after
the per-residue masked loss is aggregated at the protein level.

  Tier A (DisProt):          1.0
  Tier B (MobiDB homology):  0.7
  Tier C (PDB-only):         0.5
"""

import torch
import torch.nn.functional as F

# Label constants (must match src/data/labeling.py)
DISORDERED = 1
ORDERED = 0
MASKED = -1
AMBIGUOUS = -2

TIER_WEIGHTS = {"A": 1.0, "B": 0.7, "C": 0.5}


def build_loss_mask(labels: torch.Tensor) -> torch.Tensor:
    """Build binary mask: 1 for contributing residues, 0 for excluded.

    Contributing: DISORDERED (1) and ORDERED (0)
    Excluded: MASKED (-1) and AMBIGUOUS (-2)

    Args:
        labels: [B, L] integer label tensor.

    Returns:
        [B, L] float mask tensor (0.0 or 1.0).
    """
    return ((labels == DISORDERED) | (labels == ORDERED)).float()


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """Compute masked BCE loss over contributing residues only.

    Args:
        logits: [B, L, 1] or [B, L] — raw model output (pre-sigmoid).
        labels: [B, L] — integer labels (1, 0, -1, -2).
        pos_weight: Optional positive class weight. If None, unweighted.

    Returns:
        Scalar loss averaged over contributing residues.
    """
    # Squeeze logits to [B, L] if needed
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)

    # Build mask: only DISORDERED (1) and ORDERED (0) contribute
    mask = build_loss_mask(labels)

    # Build float targets: DISORDERED → 1.0, ORDERED → 0.0
    # Masked/ambiguous values don't matter since mask excludes them
    targets = (labels == DISORDERED).float()

    # Compute per-residue BCE loss
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        loss_per_residue = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=pw,
        )
    else:
        loss_per_residue = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )

    # Masked mean: sum over contributing residues / count of contributing
    contributing = mask.sum()
    if contributing == 0:
        # No contributing residues — return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = (loss_per_residue * mask).sum() / contributing

    assert torch.isfinite(loss), "Loss is not finite"
    return loss


# ─── Focal Loss (Amendment T-1) ──────────────────────────────────────────────


def masked_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """Compute masked focal loss over contributing residues only.

    focal_loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
      where p_t = sigmoid(logit) if target=1 else 1 - sigmoid(logit),
            alpha_t = alpha if target=1 else 1 - alpha (if alpha specified).

    At gamma=0 and alpha=None this reduces exactly to masked BCE (unit-tested).

    Args:
        logits:  [B, L, 1] or [B, L] — raw model output (pre-sigmoid).
        labels:  [B, L]             — integer labels (1, 0, -1, -2).
        gamma:   Focusing parameter (Amendment T-1: tunable 1.0–3.0, default 2.0).
        alpha:   Optional class-balance factor in (0, 1). If None, alpha_t == 1.

    Returns:
        Scalar loss averaged over contributing residues.
    """
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)

    mask = build_loss_mask(labels)
    targets = (labels == DISORDERED).float()

    # Numerically stable: use BCE(logits) as -log(p_t) for each residue.
    # -log(p_t) = BCE per-residue (this is exactly -log(p) for positives and
    # -log(1-p) for negatives, which is -log(p_t)).
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = exp(-BCE) = p for positives, 1-p for negatives.
    p_t = torch.exp(-bce).clamp(min=1e-12, max=1.0)

    focal_factor = (1.0 - p_t) ** gamma

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss_per_residue = alpha_t * focal_factor * bce
    else:
        loss_per_residue = focal_factor * bce

    contributing = mask.sum()
    if contributing == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = (loss_per_residue * mask).sum() / contributing
    assert torch.isfinite(loss), "Focal loss is not finite"
    return loss


# ─── Tiered Weighting (Amendment T-4) ────────────────────────────────────────


def per_protein_masked_loss(
    logits_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    loss_fn_name: str = "bce",
    tier_weights_per_protein: list[float] | None = None,
    **loss_kwargs,
) -> torch.Tensor:
    """Compute weighted mean of per-protein masked losses.

    For each protein in the batch:
      1. Compute masked per-residue loss (BCE or focal), averaged over that
         protein's contributing residues only (same mask logic as Part 1).
      2. Multiply by the protein's tier weight (Amendment T-4): A=1.0, B=0.7,
         C=0.5.
      3. Aggregate across proteins with weighted mean.

    Proteins with zero contributing residues contribute zero to the loss (and
    are excluded from the normalization denominator so they don't dilute it).

    Args:
        logits_list: length-N list of [L_i, 1] or [L_i] per-protein logits.
        labels_list: length-N list of [L_i] per-protein integer labels.
        loss_fn_name: "bce" or "focal".
        tier_weights_per_protein: length-N list of floats (1.0 / 0.7 / 0.5);
                                   if None, treats all weights as 1.0.
        **loss_kwargs: Passed to the underlying loss fn (pos_weight, gamma, alpha).

    Returns:
        Scalar weighted mean loss (weighted_sum / sum_of_weights_of_contributing).
    """
    if tier_weights_per_protein is None:
        tier_weights_per_protein = [1.0] * len(logits_list)
    if len(tier_weights_per_protein) != len(logits_list):
        raise ValueError(
            f"tier_weights length {len(tier_weights_per_protein)} != logits length {len(logits_list)}"
        )
    if loss_fn_name == "bce":
        loss_fn = masked_bce_loss
    elif loss_fn_name == "focal":
        loss_fn = masked_focal_loss
    else:
        raise ValueError(f"unknown loss_fn_name={loss_fn_name!r}")

    device = logits_list[0].device if logits_list else torch.device("cpu")
    weighted_loss_sum = torch.tensor(0.0, device=device)
    active_weight_sum = 0.0

    for log_i, lab_i, w_i in zip(logits_list, labels_list, tier_weights_per_protein):
        # Add batch dim so mask-builder works unchanged.
        if log_i.dim() == 1 or (log_i.dim() == 2 and log_i.size(-1) == 1):
            log_b = log_i.unsqueeze(0)
        else:
            log_b = log_i  # already has some leading dim
        lab_b = lab_i.unsqueeze(0)

        contributing = build_loss_mask(lab_b).sum().item()
        if contributing == 0:
            continue  # skip proteins that have no supervised residues

        protein_loss = loss_fn(log_b, lab_b, **loss_kwargs)
        weighted_loss_sum = weighted_loss_sum + protein_loss * w_i
        active_weight_sum += w_i

    if active_weight_sum == 0.0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return weighted_loss_sum / active_weight_sum


def masked_mil_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_disprot: torch.Tensor,
    negative_weight_disprot: float = 0.3,
) -> torch.Tensor:
    """Soft-MIL BCE: downweight ORDERED residues on DisProt proteins.

    Rationale: DisProt ORDERED labels may reflect annotation incompleteness
    (residue not yet experimentally characterized as disordered) rather than
    structural order. PDB-only proteins have structural evidence for order, so
    their ORDERED labels are kept at full weight.

    Per-residue weight:
      DISORDERED (target=1.0):                    weight = 1.0
      ORDERED on DisProt protein (target=0.0):    weight = negative_weight_disprot
      ORDERED on PDB-only protein (target=0.0):   weight = 1.0
      MASKED / AMBIGUOUS:                          weight = 0.0

    Loss is normalized by the sum of weights (effective sample size), so
    magnitude is comparable across batches with different DisProt/PDB ratios.

    Args:
        logits:                   [B, L, 1] or [B, L] — raw model output (pre-sigmoid).
        labels:                   [B, L] — integer labels (1, 0, -1, -2).
        is_disprot:               [B] — bool/0-1 tensor; True for DisProt proteins.
        negative_weight_disprot:  weight for ORDERED on DisProt (default 0.3).

    Returns:
        Scalar weighted-mean loss over contributing residues.
    """
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)

    B, L = labels.shape
    is_disprot_f = is_disprot.to(dtype=logits.dtype, device=logits.device).view(B, 1).expand(B, L)

    is_disordered = (labels == DISORDERED).to(dtype=logits.dtype)
    is_ordered = (labels == ORDERED).to(dtype=logits.dtype)

    weight = (
        is_disordered * 1.0
        + is_ordered * is_disprot_f * negative_weight_disprot
        + is_ordered * (1.0 - is_disprot_f) * 1.0
    )

    targets = is_disordered  # DISORDERED → 1.0, all others → 0.0 (mask zeros their weight)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    weight_sum = weight.sum()
    if weight_sum == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = (bce * weight).sum() / weight_sum
    assert torch.isfinite(loss), "MIL loss is not finite"
    return loss


# ─── Part 4 Phase 2 losses (Amendments T-3, T-4 ext, M-? R-Drop) ──────


def masked_label_smoothed_bce_loss(
    logits: torch.Tensor,        # [B, L, 1] or [B, L]
    labels: torch.Tensor,        # [B, L]
    smoothing: float = 0.05,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """Masked BCE with label smoothing per Amendment T-3 (Part 4).

    Targets:
        ORDERED   (0) → smoothing
        DISORDERED(1) → 1 - smoothing
    Excluded labels (AMBIGUOUS=-2, MASKED=-1) contribute zero gradient — same
    masking semantics as `masked_bce_loss`.
    """
    if not 0.0 <= smoothing < 0.5:
        raise ValueError(f"smoothing must be in [0, 0.5), got {smoothing}")
    if logits.dim() == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    mask = build_loss_mask(labels)
    targets = labels.to(logits.dtype).clamp(min=0.0, max=1.0)
    # Smooth: 0 → smoothing; 1 → 1 - smoothing
    targets = targets * (1.0 - 2.0 * smoothing) + smoothing
    pw = (
        torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None else None
    )
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pw,
    )
    per_pos = per_pos * mask.to(per_pos.dtype)
    return per_pos.sum() / mask.sum().clamp_min(1).to(per_pos.dtype)


def masked_asymmetric_bce_loss(
    logits: torch.Tensor,        # [B, L, 1] or [B, L]
    labels: torch.Tensor,        # [B, L]
    fp_weight: float = 1.5,
    fn_weight: float = 1.0,
) -> torch.Tensor:
    """Masked BCE with asymmetric per-error weighting per Amendment T-4-ext.

    Per-residue weighting rule (applied AFTER computing the standard BCE term,
    before averaging):
        - label=0 with predicted prob > 0.5  → false positive  → fp_weight
        - label=1 with predicted prob < 0.5  → false negative  → fn_weight
        - all other (correct or borderline)  → 1.0

    The threshold is fixed at 0.5 to match the conventional FP/FN definition.
    Excluded labels (AMBIGUOUS=-2, MASKED=-1) contribute zero gradient.

    Default fp_weight = 1.5 (P4_04 §S01) — pushes the model to be more
    conservative on potential ordered residues, targeting the NOX precision
    failure mode diagnosed in Parts 1–3.
    """
    if logits.dim() == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    mask = build_loss_mask(labels)
    targets = labels.to(logits.dtype).clamp(min=0.0, max=1.0)
    per_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        is_fp = (labels == ORDERED) & (probs > 0.5)
        is_fn = (labels == DISORDERED) & (probs < 0.5)
        weight = torch.ones_like(per_pos)
        weight = torch.where(is_fp, torch.full_like(weight, fp_weight), weight)
        weight = torch.where(is_fn, torch.full_like(weight, fn_weight), weight)
    per_pos = per_pos * weight * mask.to(per_pos.dtype)
    return per_pos.sum() / mask.sum().clamp_min(1).to(per_pos.dtype)


def rdrop_kl_consistency_loss(
    logits_a: torch.Tensor,      # [B, L, 1] or [B, L]
    logits_b: torch.Tensor,      # same shape — second forward with different dropout
    labels: torch.Tensor,        # [B, L]
) -> torch.Tensor:
    """Symmetric KL divergence between two dropout-perturbed forward passes
    of the same batch (R-Drop, Liang et al. 2021).

    Combined R-Drop training loop:
        loss = 0.5 * (bce(a) + bce(b)) + alpha * rdrop_kl_consistency_loss(a, b, labels)

    Default alpha = 0.5 (P4_04 §S01). Adds ~50% training time vs single-pass.
    Mask: residues with excluded labels do not contribute to KL.
    """
    if logits_a.dim() == 3 and logits_a.shape[-1] == 1:
        logits_a = logits_a.squeeze(-1)
    if logits_b.dim() == 3 and logits_b.shape[-1] == 1:
        logits_b = logits_b.squeeze(-1)
    mask = build_loss_mask(labels).to(logits_a.dtype)
    # Build 2-class log-prob distributions [p, 1-p] for each pass and KL them
    log_pa = torch.nn.functional.logsigmoid(logits_a)
    log_pa1 = torch.nn.functional.logsigmoid(-logits_a)
    log_pb = torch.nn.functional.logsigmoid(logits_b)
    log_pb1 = torch.nn.functional.logsigmoid(-logits_b)
    pa = torch.exp(log_pa); pa1 = torch.exp(log_pa1)
    pb = torch.exp(log_pb); pb1 = torch.exp(log_pb1)
    kl_ab = pa * (log_pa - log_pb) + pa1 * (log_pa1 - log_pb1)
    kl_ba = pb * (log_pb - log_pa) + pb1 * (log_pb1 - log_pa1)
    sym_kl = 0.5 * (kl_ab + kl_ba) * mask
    return sym_kl.sum() / mask.sum().clamp_min(1.0)


def tier_weights_from_sources(sources: list[str]) -> list[float]:
    """Map source tags to tier weights per Amendment T-4.

    Accepts either source strings ('disprot', 'mobidb_homology', 'pdb_only')
    or tier letters ('A', 'B', 'C').
    """
    SOURCE_TO_TIER = {
        "disprot": "A",
        "mobidb_homology": "B",
        "pdb_only": "C",
        "A": "A", "B": "B", "C": "C",
    }
    weights = []
    for s in sources:
        tier = SOURCE_TO_TIER.get(s)
        if tier is None:
            raise ValueError(f"unknown source/tier tag: {s!r}")
        weights.append(TIER_WEIGHTS[tier])
    return weights
