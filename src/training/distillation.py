"""Knowledge distillation from Part 4 ensemble (Part 5 Phase 2 EXP-T04).

Reference: P5_04 §P5-Ph2-S03; CLAUDE.md rule 31 (T=3-5 required).

Mechanism:
  - Teacher: Part 4 C-deep5+TTA per-residue probabilities, FROZEN, pre-cached.
  - Student: SaProt E13H trained from scratch.
  - Temperature-scaled soft targets:
        teacher_soft = sigmoid(teacher_logit / T)
        student_soft = sigmoid(student_logit / T)
  - Combined loss:
        L = alpha * task_BCE(student_logit, hard_label)
          + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)
  - Default T=4, alpha=0.5.

Note: teacher predictions in the cache (features/self_refinement/part4_predictions.h5)
are stored as PROBABILITIES, not logits. We invert sigmoid to recover logits
via: logit = log(p / (1 - p)) with epsilon clamping for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DISORDERED = 1
ORDERED = 0


def prob_to_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Invert sigmoid: logit = log(p / (1 - p)).

    Used to recover teacher logits from cached probabilities.
    """
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


def distillation_kl_loss(
    student_logits: torch.Tensor,    # [B, L, 1] or [B, L]
    teacher_logits: torch.Tensor,    # [B, L]    same shape (post-squeeze)
    labels: torch.Tensor,             # [B, L]   for masking
    T: float = 4.0,
) -> torch.Tensor:
    """KL divergence between student and teacher soft predictions at temp T.

    KL(student || teacher) for Bernoulli:
        p_s * log(p_s / p_t) + (1 - p_s) * log((1 - p_s) / (1 - p_t))

    Masked to supervised residues (DISORDERED or ORDERED).
    Multiplied by T^2 by the caller (Hinton convention).
    """
    if student_logits.dim() == 3 and student_logits.size(-1) == 1:
        student_logits = student_logits.squeeze(-1)
    if teacher_logits.dim() == 3 and teacher_logits.size(-1) == 1:
        teacher_logits = teacher_logits.squeeze(-1)

    mask = ((labels == DISORDERED) | (labels == ORDERED)).float()

    # Soft probs at temperature T
    log_ps = F.logsigmoid(student_logits / T)
    log_ps1 = F.logsigmoid(-student_logits / T)
    with torch.no_grad():
        log_pt = F.logsigmoid(teacher_logits / T)
        log_pt1 = F.logsigmoid(-teacher_logits / T)

    ps = torch.exp(log_ps)
    ps1 = torch.exp(log_ps1)

    kl = ps * (log_ps - log_pt) + ps1 * (log_ps1 - log_pt1)  # [B, L]
    kl = kl * mask
    denom = mask.sum().clamp_min(1.0)
    return kl.sum() / denom


def combined_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    T: float = 4.0,
    task_loss_fn=None,
    task_loss_kwargs: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combined distillation loss:
        L = alpha * task_BCE + (1 - alpha) * T^2 * KL(student || teacher)

    Args:
        student_logits: [B, L, 1] or [B, L]
        teacher_logits: [B, L] (caller is responsible for inverting sigmoid)
        labels:         [B, L] integer
        alpha:          task vs distill weight (0.5 default)
        T:              softmax temperature (4 default)
        task_loss_fn:   the BCE/focal/etc. loss callable (signature(logits, labels, **kw))
        task_loss_kwargs: dict of extra args for task_loss_fn

    Returns:
        (total_loss, diag) where diag has 'task_loss' and 'distill_loss' (raw, no T^2).
    """
    if task_loss_fn is None:
        from src.training.loss import masked_bce_loss as task_loss_fn  # noqa: PLC0415
    task_kwargs = task_loss_kwargs or {}
    task_loss = task_loss_fn(student_logits, labels, **task_kwargs)

    distill_loss = distillation_kl_loss(student_logits, teacher_logits, labels, T=T)

    total = alpha * task_loss + (1.0 - alpha) * (T * T) * distill_loss
    diag = {
        "task_loss": float(task_loss.detach().item()),
        "distill_loss": float(distill_loss.detach().item()),
        "T": T,
        "alpha": alpha,
    }
    return total, diag


def prediction_entropy(probs: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean Bernoulli entropy of supervised residues (for diagnostic).

    H(p) = -p*log(p) - (1-p)*log(1-p), in nats.
    """
    p = probs.clamp(min=1e-8, max=1.0 - 1e-8)
    h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    h = h * mask
    denom = mask.sum().clamp_min(1.0)
    return float((h.sum() / denom).item())
