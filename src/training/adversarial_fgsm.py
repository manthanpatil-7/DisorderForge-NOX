"""Adversarial FGSM training on protein embeddings (Part 5 Phase 2 EXP-T05).

Reference: P5_04 §P5-Ph2-S04; CLAUDE.md rule 30 (epsilon: 0.01-0.1, never above);
CLAUDE.md gotcha (FGSM applied to embeddings, not raw sequences).

Mechanism:
  - For each batch: forward on clean input -> clean_loss.
  - Compute gradient of clean_loss w.r.t. INPUT EMBEDDINGS.
  - Perturb: emb_adv = emb + epsilon * sign(grad).
  - Forward on perturbed input -> adv_loss.
  - Total: 0.5 * clean_loss + 0.5 * adv_loss.
  - Backward and step.

Key implementation details:
  - We do NOT use the model's internal grad for FGSM; we compute it explicitly
    via torch.autograd.grad to keep the adversarial perturbation crisp.
  - The self-refinement channel must NOT be perturbed (it is FROZEN per
    CLAUDE.md rule 27). The training pipeline tags channel slices, and we
    zero out the perturbation on the SR slice.
  - Epsilon defaults to 0.01, capped at 0.05 (well below CLAUDE.md ceiling 0.1).
"""

from __future__ import annotations

import torch


def fgsm_perturb_embeddings(
    features: torch.Tensor,                # [B, L, F]
    loss: torch.Tensor,                     # scalar — clean loss
    epsilon: float = 0.01,
    sr_slice: tuple[int, int] | None = None,  # if set, perturbation is zero in [s, e)
) -> torch.Tensor:
    """Compute FGSM perturbation given the clean-pass loss.

    Returns:
        delta: [B, L, F] perturbation tensor (no_grad context applied).

    The caller is responsible for calling features.requires_grad_(True) before
    the clean forward pass (or detaching/cloning as needed afterward).
    """
    grad = torch.autograd.grad(
        outputs=loss, inputs=features,
        retain_graph=True, create_graph=False, only_inputs=True,
    )[0]
    delta = epsilon * grad.sign().detach()
    if sr_slice is not None:
        s, e = sr_slice
        delta[..., s:e] = 0.0  # do not perturb FROZEN self-refinement channel
    return delta


def fgsm_step_loss(
    model,
    features: torch.Tensor,                # [B, L, F]
    labels: torch.Tensor,                   # [B, L]
    forward_fn,                             # callable(model, features) -> dict with 'logit_disorder'
    task_loss_fn,                            # callable(logits, labels, **kw) -> loss
    task_loss_kwargs: dict | None = None,
    epsilon: float = 0.01,
    clean_weight: float = 0.5,
    adv_weight: float = 0.5,
    sr_slice: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict]:
    """One FGSM training step.

    Args:
        model:              torch.nn.Module
        features:           [B, L, F]
        labels:             [B, L]
        forward_fn:         takes (model, features) and returns the dict the model
                            usually emits (must include 'logit_disorder' as [B, L, 1]).
        task_loss_fn:       BCE/etc. callable
        task_loss_kwargs:   extra kwargs for task_loss_fn
        epsilon:            FGSM step size (default 0.01, cap 0.05)
        clean_weight:       coefficient for clean loss (default 0.5)
        adv_weight:         coefficient for adversarial loss (default 0.5)
        sr_slice:           (s, e) tuple — channel range NOT to perturb (frozen SR)

    Returns:
        (total_loss, diag)
    """
    if epsilon > 0.05:
        raise ValueError(f"FGSM epsilon {epsilon} exceeds safety cap 0.05")
    task_kwargs = task_loss_kwargs or {}

    # Clean pass with grad-tracked embeddings
    feats_clean = features.detach().clone().requires_grad_(True)
    out_clean = forward_fn(model, feats_clean)
    logits_clean = out_clean["logit_disorder"]
    clean_loss = task_loss_fn(logits_clean, labels, **task_kwargs)

    # Compute FGSM perturbation
    with torch.enable_grad():
        delta = fgsm_perturb_embeddings(
            feats_clean, clean_loss, epsilon=epsilon, sr_slice=sr_slice,
        )

    # Adversarial pass
    feats_adv = (features.detach() + delta).requires_grad_(False)
    out_adv = forward_fn(model, feats_adv)
    logits_adv = out_adv["logit_disorder"]
    adv_loss = task_loss_fn(logits_adv, labels, **task_kwargs)

    total = clean_weight * clean_loss + adv_weight * adv_loss

    # Diagnostic: clean vs adversarial accuracy
    with torch.no_grad():
        from src.training.loss import build_loss_mask  # noqa: PLC0415
        mask = build_loss_mask(labels)
        targets = (labels == 1).float()

        def _acc(logits):
            if logits.dim() == 3 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = ((preds == targets) * mask).sum()
            denom = mask.sum().clamp_min(1.0)
            return float((correct / denom).item())

        diag = {
            "clean_loss": float(clean_loss.detach().item()),
            "adv_loss": float(adv_loss.detach().item()),
            "clean_acc": _acc(logits_clean.detach()),
            "adv_acc": _acc(logits_adv.detach()),
            "epsilon": epsilon,
            "delta_l_inf": float(delta.abs().max().item()),
        }
    return total, diag
